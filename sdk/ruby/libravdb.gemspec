Gem::Specification.new do |spec|
  spec.name          = "libravdb"
  spec.version       = "1.0.0"
  spec.authors       = ["LibraVDB Team"]
  spec.summary       = "Ruby SDK for LibraVDB"
  spec.description   = "High performance FFI bindings to the LibraVDB C-Shared library."
  spec.homepage      = "https://github.com/xDarkicex/libraVDB"
  spec.license       = "MIT"

  spec.files         = Dir["lib/**/*"] + Dir["ext/**/*"]
  spec.require_paths = ["lib"]

  spec.add_dependency "ffi", "~> 1.15.0"
  spec.add_dependency "json"
end
