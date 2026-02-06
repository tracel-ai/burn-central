cargo run --example axum_client -- --prompt "what is your favorite food" &
pid1=$!
cargo run --example axum_client -- --prompt "what is the capital of france" &
pid2=$!
cargo run --example axum_client -- --prompt "explain machine learning" &
pid3=$!
cargo run --example axum_client -- --prompt "write a hello world program" &
pid4=$!

wait $pid1 $pid2 $pid3 $pid4

echo "All tasks completed"