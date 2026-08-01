// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "LibraVDB",
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "LibraVDB",
            targets: ["LibraVDB"]),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "CLibraVDB",
            path: "Sources/CLibraVDB",
            publicHeadersPath: "include"
        ),
        .target(
            name: "LibraVDB",
            dependencies: ["CLibraVDB"],
            linkerSettings: [
                .unsafeFlags(["-L../cgo", "-lravdb", "-Xlinker", "-rpath", "-Xlinker", "../cgo"])
            ]
        ),
        .testTarget(
            name: "LibraVDBTests",
            dependencies: ["LibraVDB"],
            linkerSettings: [
                .unsafeFlags(["-L../cgo", "-lravdb", "-Xlinker", "-rpath", "-Xlinker", "../cgo"])
            ]
        ),
    ]
)
