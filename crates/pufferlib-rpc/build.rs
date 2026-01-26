fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check if protoc is installed
    if std::process::Command::new("protoc").arg("--version").output().is_err() {
        println!("cargo:warning=protoc not found in PATH. Please install Protocol Buffers compiler.");
        println!("cargo:warning=Windows: winget install -e --id ProtocolBuffers.Protoc");
        println!("cargo:warning=Ubuntu: sudo apt install protobuf-compiler");
        println!("cargo:warning=MacOS: brew install protobuf");
        // We can't proceed without protoc for this crate
        // Returning Err here will stop the build with a descriptive message
        return Err("protoc is required to build pufferlib-rpc. See warnings for installation instructions.".into());
    }

    tonic_build::compile_protos("proto/remote_env.proto")?;
    Ok(())
}
