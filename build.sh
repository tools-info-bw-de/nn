# build wasm
cd go && GOOS=js GOARCH=wasm go build -o nn.wasm

# build frontend
cd ../frontend && npm run build