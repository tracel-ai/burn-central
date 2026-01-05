//! # Burn Central Macros
//! As define in the burn central crate documentation, this crate provide the macros to
//! register a functions. You probably don't need more information to it then that, but if you
//! do we got you covered.
//!
//! ## Role and responsablity
//! The role of the macros is really light weight. We don't want to make the register crate the
//! hearth of our runtime. So it simply wrap your function into another functions define in the
//! runtime and the runtime does the rest of the magic.
//!
//! ## Usage
//! To use the macros you simply need to import it from this crate and use the `register` macro
//! to mark your training and inference functions. Here is an example:
//! ```ignore
//! # use burn_central_macros::register;
//! #[register(training, name = "my_training_procedure")]
//! async fn my_training_function() {
//!  // Your training code here
//! }
//! ```

use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;

use strum::Display;
use syn::{Error, ItemFn, Meta, Path, parse_macro_input, punctuated::Punctuated, spanned::Spanned};

#[derive(Eq, Hash, PartialEq, Display)]
#[strum(serialize_all = "PascalCase")]
enum ProcedureType {
    Training,
    Inference,
}

impl TryFrom<Path> for ProcedureType {
    type Error = Error;

    fn try_from(path: Path) -> Result<Self, Self::Error> {
        match path.get_ident() {
            Some(ident) => match ident.to_string().as_str() {
                "training" => Ok(Self::Training),
                "inference" => Ok(Self::Inference),
                _ => Err(Error::new_spanned(
                    path,
                    "Expected `training` or `inference`",
                )),
            },
            None => Err(Error::new_spanned(
                path,
                "Expected `training` or `inference`",
            )),
        }
    }
}

fn compile_errors(errors: Vec<Error>) -> proc_macro2::TokenStream {
    errors
        .into_iter()
        .map(|err| err.to_compile_error())
        .collect()
}

fn generate_flag_register_stream(
    item: &ItemFn,
    builder_fn_ident: &Ident,
    procedure_type: &ProcedureType,
    routine_name: &syn::LitStr,
) -> proc_macro2::TokenStream {
    let fn_name = &item.sig.ident;
    let builder_fn_name = &builder_fn_ident;
    let proc_type_str = procedure_type.to_string().to_lowercase();

    let serialized_fn_item = syn_serde::json::to_string(item);
    let serialized_bytes = serialized_fn_item.as_bytes();
    let byte_array_literal = syn::LitByteStr::new(serialized_bytes, item.span());

    let const_name = Ident::new(
        &format!(
            "BURN_CENTRAL_FUNCTION_{}",
            fn_name.to_string().to_uppercase()
        ),
        proc_macro2::Span::call_site(),
    );

    let ast_const_name = Ident::new(
        &format!("_BURN_FUNCTION_AST_{}", fn_name.to_string().to_uppercase()),
        proc_macro2::Span::call_site(),
    );

    quote! {
        const _: () = {
            #[allow(dead_code)]
            const #const_name: &str = concat!(
                "BCFN1|",
                module_path!(),
                "|",
                stringify!(#fn_name),
                "|",
                stringify!(#builder_fn_name),
                "|",
                #routine_name,
                "|",
                #proc_type_str,
                "|END"
            );

            #[allow(dead_code)]
            const #ast_const_name: &[u8] = #byte_array_literal;
        };
    }
}

fn get_string_arg(
    args: &Punctuated<Meta, syn::Token![,]>,
    arg_name: &str,
    errors: &mut Vec<Error>,
) -> Option<syn::LitStr> {
    args.iter()
        .find(|meta| meta.path().is_ident(arg_name))
        .and_then(|meta| match meta.require_name_value() {
            Ok(value) => match &value.value {
                syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(name),
                    ..
                }) => Some(name.clone()),
                _ => {
                    errors.push(Error::new(
                        value.value.span(),
                        format!("Expected a string literal for the `{arg_name}` argument."),
                    ));
                    None
                }
            },
            Err(err) => {
                errors.push(err);
                None
            }
        })
}

fn validate_registered_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("Registered name cannot be empty.".to_string());
    }
    if name.contains(' ') {
        return Err("Registered name cannot contain spaces.".to_string());
    }
    if !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return Err(
            "Registered name can only contain alphanumeric characters and underscores.".to_string(),
        );
    }
    Ok(())
}

#[proc_macro_attribute]
/// Macro to register your training and inference functions.
pub fn register(args: TokenStream, item: TokenStream) -> TokenStream {
    let mut errors = Vec::<Error>::new();
    let args = parse_macro_input!(args with Punctuated::<Meta, syn::Token![,]>::parse_terminated);
    let item = parse_macro_input!(item as ItemFn);
    let fn_name = &item.sig.ident;

    if args.is_empty() {
        errors.push(Error::new(
            args.span(),
            "Expected one argument for the #[register] attribute. Please provide the procedure type (training or inference) as the first argument.",
        ));
    }

    // Determine the proc type (training or inference)
    let procedure_type = match ProcedureType::try_from(
        args.first()
            .expect("Should be able to get first arg.")
            .path()
            .clone(),
    ) {
        Ok(procedure_type) => procedure_type,
        Err(err) => {
            return err.into_compile_error().into();
        }
    };

    if procedure_type == ProcedureType::Inference {
        errors.push(Error::new_spanned(
            args.first().unwrap().path(),
            "Inference procedures are not supported yet. Please use training procedures.",
        ));
    }

    let maybe_registered_name = get_string_arg(&args, "name", &mut errors);

    if let Some(name) = &maybe_registered_name {
        if let Err(err) = validate_registered_name(&name.value()) {
            errors.push(Error::new_spanned(
                name,
                format!("Invalid registered name: {err}"),
            ));
        }
    }

    let builder_fn_name = syn::Ident::new(
        &format!("__{fn_name}_builder"),
        proc_macro2::Span::call_site(),
    );

    let registered_name_str = {
        let name = maybe_registered_name
            .map(|name| name.value())
            .unwrap_or_else(|| fn_name.to_string());

        syn::LitStr::new(&name, fn_name.span())
    };

    let builder_item = match procedure_type {
        ProcedureType::Training => {
            quote! {
                #[doc(hidden)]
                pub fn #builder_fn_name<B: burn::tensor::backend::AutodiffBackend>(
                    exec: &mut burn_central::runtime::ExecutorBuilder<B>,
                ) {
                    exec.train(#registered_name_str, #fn_name);
                }
            }
        }
        ProcedureType::Inference => {
            quote! {}
        }
    };

    let flag_register = generate_flag_register_stream(
        &item,
        &builder_fn_name,
        &procedure_type,
        &registered_name_str,
    );

    let code = quote! {
        #[allow(dead_code)]
        #item

        #flag_register
        #builder_item
    };

    // If there are any errors, combine them and return
    if !errors.is_empty() {
        compile_errors(errors).into()
    } else {
        code.into()
    }
}

/// Macro to create an inference handler with a convenient builder module.
///
/// This macro transforms an inference handler function into a module with a `build` function
/// that makes it easy to construct an `Inference` instance.
///
/// # Example
///
/// ```ignore
/// use burn_central_runtime::inference::{In, ModelAccessor, OutStream, CancelToken};
///
/// #[inference_handler]
/// fn my_handler<B: Backend>(
///     In(input): In<String>,
///     model: ModelAccessor<MyModel<B>>,
///     output: OutStream<String>,
/// ) -> Result<(), String> {
///     // handler logic
/// }
///
/// // Now you can use:
/// let inference = my_handler::build(model);
/// ```
#[proc_macro_attribute]
pub fn inference_handler(_args: TokenStream, item: TokenStream) -> TokenStream {
    let item = parse_macro_input!(item as ItemFn);
    let fn_name = &item.sig.ident;
    let fn_vis = &item.vis;

    // Extract type parameters from the function
    let generics = &item.sig.generics;
    let where_clause = &generics.where_clause;

    // Extract the model type from ModelAccessor parameter
    let mut model_type = None;
    let mut backend_param = None;

    for param in &item.sig.inputs {
        if let syn::FnArg::Typed(pat_type) = param {
            if let syn::Type::Path(type_path) = &*pat_type.ty {
                let path = &type_path.path;
                if let Some(segment) = path.segments.last() {
                    if segment.ident == "ModelAccessor" {
                        if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                            if let Some(syn::GenericArgument::Type(ty)) = args.args.first() {
                                model_type = Some(ty.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    // Find the Backend type parameter
    if let Some(generics) = generics.params.first() {
        if let syn::GenericParam::Type(type_param) = generics {
            backend_param = Some(type_param.ident.clone());
        }
    }

    let model_ty = model_type.expect("ModelAccessor parameter not found");
    let backend = backend_param.unwrap_or_else(|| syn::parse_quote!(B));

    let mod_name = fn_name;

    // Build a combined where clause that includes Send requirements
    let combined_where_clause = if let Some(wc) = where_clause {
        quote! {
            #wc, #model_ty: Send + 'static
        }
    } else {
        quote! {
            where #model_ty: Send + 'static
        }
    };

    let code = quote! {
        // The actual handler function
        #item

        // Module with the same name as the handler
        #fn_vis mod #mod_name {
            use super::*;

            /// Create a builder pre-loaded with the model, ready for you to call `.build(handler)`.
            ///
            /// # Arguments
            /// * `model` - The model to use for inference
            ///
            /// # Example
            /// ```ignore
            /// let inference = my_handler::builder(model).build(my_handler);
            /// ```
            pub fn build #generics (
                model: #model_ty,
            ) -> burn_central_runtime::inference::Inference<#backend, impl Send + 'static, burn_central_runtime::inference::In<impl Send + 'static>, impl Send + 'static>
            #combined_where_clause
            {
                let model: Box<dyn std::any::Any + Send> = Box::new(model);
                let #fn_name: Box<dyn std::any::Any> = Box::new(#fn_name);
                burn_central_runtime::inference::InferenceBuilder::<#backend>::new()
                    .with_model(*model.downcast().expect("Failed to downcast model"))
                    .build(
                        *#fn_name
                            .downcast()
                            .expect("Failed to downcast inference handler function"),
                    )
            }
        }
    };

    code.into()
}
