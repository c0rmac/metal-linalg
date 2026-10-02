// Tests of the MLXArray layer: shapes, values and batch dimensions.
//
// MLX itself runs on its CPU here, so that the checks do not depend on how
// MLX's own GPU kernels were built: `swift build` cannot compile them, and a
// metallib borrowed from another MLX build may not match (one from MLX
// 0.32.1 under mlx-swift's 0.32.2 gave matmul errors of 4e-3). metal-linalg's
// kernels are embedded and run on the GPU regardless. MLX still loads a
// metallib at start-up; see docs/swift.md for running these from the command line.
import MLX
import XCTest

@testable import MetalLinalgMLX

final class MetalLinalgMLXTests: XCTestCase {
    func onCPU(_ body: () throws -> Void) rethrows {
        try Device.withDefaultDevice(.cpu, body)
    }

    func maxAbs(_ x: MLXArray) -> Float { abs(x).max().item(Float.self) }

    func testQR() throws {
        try onCPU {
            let a = MLXRandom.normal([4, 3, 20, 8])     // two batch dimensions
            let (q, r) = try qrAccelerated(a)
            XCTAssertEqual(q.shape, [4, 3, 20, 8])
            XCTAssertEqual(r.shape, [4, 3, 8, 8])
            XCTAssertLessThan(maxAbs(matmul(q, r) - a), 1e-4)
        }
    }

    func testEigh() throws {
        try onCPU {
            let x = MLXRandom.normal([64, 12, 12])
            let s = x + x.transposed(0, 2, 1)
            let (w, v) = try eighAccelerated(s)
            XCTAssertEqual(w.shape, [64, 12])
            XCTAssertEqual(v.shape, [64, 12, 12])
            XCTAssertLessThan(maxAbs(matmul(s, v) - v * w.expandedDimensions(axis: 1)), 1e-3)
            XCTAssertLessThan(maxAbs(try eigvalshAccelerated(s, uplo: .upper) - w), 1e-4)
        }
    }

    func testSVD() throws {
        try onCPU {
            for shape in [[8, 30, 10], [8, 10, 30]] {
                let a = MLXRandom.normal(shape)
                let k = min(shape[1], shape[2])
                let (u, s, vt) = try svdAccelerated(a)
                XCTAssertEqual(u.shape, [8, shape[1], k])
                XCTAssertEqual(s.shape, [8, k])
                XCTAssertEqual(vt.shape, [8, k, shape[2]])
                XCTAssertLessThan(maxAbs(matmul(u * s.expandedDimensions(axis: 1), vt) - a), 1e-4)
                XCTAssertLessThan(maxAbs(try svdvalsAccelerated(a) - s), 1e-4)
            }
        }
    }

    func testErrors() throws {
        try onCPU {
            XCTAssertThrowsError(try qrAccelerated(MLXArray([1, 2, 3] as [Float])))   // 1-D
            XCTAssertThrowsError(try eighAccelerated(MLXRandom.normal([3, 4])))      // not square
            XCTAssertThrowsError(try svdAccelerated(MLXArray.eye(3).asType(.complex64))) // complex
        }
    }
}
