// swift-tools-version:5.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "swift-linear-regression",
  dependencies: [
    // .package(url: "https://github.com/tensorflow/swift-apis.git", from: "v0.2")
  ],
  targets: [
    // Targets are the basic building blocks of a package. A target can define a module or a test suite.
    // Targets can depend on other targets in this package, and on products in packages this package depends on.
    .target(
      name: "swift-linear-regression",
      dependencies: [])
//    .testTarget(
//      name: "swift-linear-regressionTests",
//      dependencies: ["swift-linear-regression"])
  ]
)
