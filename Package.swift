// swift-tools-version: 5.9
// The Swift package: the buffer core compiled from src/, the C API as the
// module Swift imports, a Swift API on [Float] (MetalLinalg), and MLXArray
// overloads (MetalLinalgMLX) on mlx-swift. See docs/swift.md.
//
// METAL_LINALG_NO_MLX=1 leaves out MetalLinalgMLX and the mlx-swift
// dependency, to build and test the rest without fetching MLX.
import Foundation
import PackageDescription

let withMLX = ProcessInfo.processInfo.environment["METAL_LINALG_NO_MLX"] == nil

let coreSources = [
    "src/device.mm",
    "src/metal_runtime.mm",
    "src/qr.mm",
    "src/qr_unblocked.mm",
    "src/qr_streaming_amx_reduced.mm",
    "src/qr_streaming_amx_complete.mm",
    "src/qr_cpu.mm",
    "src/eigh.mm",
    "src/eigh_block_jacobi.mm",
    "src/svd.mm",
    "src/svd_block_jacobi.mm",
    "src/c_api.cpp",
    "swift/CMetalLinalg/embedded_shaders.c",
]

// The core target's directory is the repository root, so that it compiles
// src/ and include/ in place. SwiftPM treats some files under a target's
// directory as resources whatever its source list, .metal among them, so
// everything else is excluded: the shaders reach the package compiled, from
// shaders/prebuilt/ through swift/CMetalLinalg/embedded_shaders.c.
let notCore = [
    ".github", "CHANGELOG.md", "CMakeLists.txt", "README.md", "benchmarks", "cmake", "docs",
    "examples", "python", "pyproject.toml", "shaders", "tests", "tuning", "src/tuned",
    "swift/Sources", "swift/Tests",
]

var products: [Product] = [
    .library(name: "MetalLinalg", targets: ["MetalLinalg"]),
]
var dependencies: [Package.Dependency] = []
var targets: [Target] = [
    .target(
        name: "CMetalLinalg",
        path: ".",
        exclude: notCore,
        sources: coreSources,
        publicHeadersPath: "swift/CMetalLinalg/include",
        cSettings: [.headerSearchPath("include"), .headerSearchPath("src")],
        cxxSettings: [.headerSearchPath("include"), .headerSearchPath("src")],
        linkerSettings: [
            .linkedFramework("Metal"),
            .linkedFramework("MetalPerformanceShaders"),
            .linkedFramework("Foundation"),
            .linkedFramework("IOKit"),
            .linkedFramework("Accelerate"),
        ]
    ),
    .target(name: "MetalLinalg", dependencies: ["CMetalLinalg"], path: "swift/Sources/MetalLinalg"),
    .testTarget(name: "MetalLinalgTests", dependencies: ["MetalLinalg"], path: "swift/Tests/MetalLinalgTests"),
]

if withMLX {
    products.append(.library(name: "MetalLinalgMLX", targets: ["MetalLinalgMLX"]))
    dependencies.append(.package(url: "https://github.com/ml-explore/mlx-swift", from: "0.25.0"))
    targets.append(.target(
        name: "MetalLinalgMLX",
        dependencies: ["MetalLinalg", .product(name: "MLX", package: "mlx-swift")],
        path: "swift/Sources/MetalLinalgMLX"
    ))
    targets.append(.testTarget(
        name: "MetalLinalgMLXTests",
        dependencies: ["MetalLinalgMLX", .product(name: "MLX", package: "mlx-swift")],
        path: "swift/Tests/MetalLinalgMLXTests"
    ))
}

let package = Package(
    name: "metal-linalg",
    platforms: [.macOS(.v14)],
    products: products,
    dependencies: dependencies,
    targets: targets,
    cxxLanguageStandard: .cxx20
)
