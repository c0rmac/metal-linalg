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

    func testQRModes() throws {
        try onCPU {
            let a = MLXRandom.normal([6, 24, 10])
            let (_, r) = try qrAccelerated(a)
            let (q0, rAlone) = try qrAccelerated(a, mode: .r)
            XCTAssertEqual(q0.shape, [0])
            XCTAssertEqual(maxAbs(rAlone - r), 0)
            let (qc, rc) = try qrAccelerated(a, mode: .complete)
            XCTAssertEqual(qc.shape, [6, 24, 24])
            XCTAssertEqual(rc.shape, [6, 24, 10])
            XCTAssertLessThan(maxAbs(matmul(qc, rc) - a), 1e-4)
            XCTAssertLessThan(maxAbs(matmul(qc.transposed(0, 2, 1), qc) - MLXArray.identity(24)), 1e-5)
            let (qi, ri) = try qrAccelerated(MLXArray.zeros([2, 4, 0]), mode: .complete)   // nothing to factor
            XCTAssertEqual(qi.shape, [2, 4, 4])
            XCTAssertEqual(ri.shape, [2, 4, 0])
            XCTAssertEqual(maxAbs(qi - MLXArray.identity(4)), 0)
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

    func testCholesky() throws {
        try onCPU {
            let x = MLXRandom.normal([2, 3, 16, 16])     // two batch dimensions
            let p = matmul(x, x.transposed(0, 1, 3, 2)) / 16 + MLXArray.identity(16)
            let l = try choleskyAccelerated(p)
            XCTAssertEqual(l.shape, [2, 3, 16, 16])
            XCTAssertLessThan(maxAbs(matmul(l, l.transposed(0, 1, 3, 2)) - p), 1e-4)
            XCTAssertEqual(maxAbs(triu(l, k: 1)), 0)
            let u = try choleskyAccelerated(p, upper: true)
            XCTAssertLessThan(maxAbs(u - l.transposed(0, 1, 3, 2)), 1e-5)
        }
    }

    func testSolveAndInverse() throws {
        try onCPU {
            let a = MLXRandom.normal([3, 20, 20]) + 8 * MLXArray.identity(20)
            let b = MLXRandom.normal([3, 20, 4])
            let x = try solveAccelerated(a, b)
            XCTAssertEqual(x.shape, [3, 20, 4])
            XCTAssertLessThan(maxAbs(matmul(a, x) - b), 1e-4)
            let inverse = try invAccelerated(a)
            XCTAssertLessThan(maxAbs(matmul(a, inverse) - MLXArray.identity(20)), 1e-4)
            let l = tril(a)
            let y = try solveTriangularAccelerated(l, b)
            XCTAssertLessThan(maxAbs(matmul(l, y) - b), 1e-4)
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

    // The input is read where MLX keeps it: a strided view or another dtype
    // must arrive as the matrix it stands for, and the input must be left as
    // it was.
    func testViewsAndDtypes() throws {
        try onCPU {
            let base = MLXRandom.normal([6, 12, 20])
            let before = base * 1
            for a in [base.transposed(0, 2, 1),                // strided
                      base[0..., 2..., 0 ..< 8],                // an offset and strides
                      base.asType(.float16)] {                 // converted
                let (q, r) = try qrAccelerated(a)
                XCTAssertEqual(q.dtype, .float32)
                XCTAssertLessThan(maxAbs(matmul(q, r) - a.asType(.float32)), 1e-3)
            }
            XCTAssertEqual(maxAbs(base - before), 0)
        }
    }

    // Results are memory the output arrays own: they outlive the input and
    // one another, and many calls run without the memory going astray.
    func testOutputsOwnTheirMemory() throws {
        try onCPU {
            var results: [(MLXArray, MLXArray, MLXArray)] = []
            for _ in 0 ..< 200 {
                let a = MLXRandom.normal([3, 7, 5])
                let (q, r) = try qrAccelerated(a)
                results.append((a * 1, q, r))
            }
            for (a, q, r) in results.suffix(20) {
                XCTAssertLessThan(maxAbs(matmul(q, r) - a), 1e-4)
            }
        }
    }

    func testEmpty() throws {
        try onCPU {
            let (q, r) = try qrAccelerated(MLXArray.zeros([0, 5, 3]))
            XCTAssertEqual(q.shape, [0, 5, 3])
            XCTAssertEqual(r.shape, [0, 3, 3])
            let (w, v) = try eighAccelerated(MLXArray.zeros([2, 0, 0]))
            XCTAssertEqual(w.shape, [2, 0])
            XCTAssertEqual(v.shape, [2, 0, 0])
            XCTAssertEqual(try svdvalsAccelerated(MLXArray.zeros([4, 6, 0])).shape, [4, 0])
        }
    }

    // A batch large enough for the GPU on a measured Mac.
    func testLargeBatch() throws {
        try onCPU {
            let x = MLXRandom.normal([4096, 32, 32])
            let s = x + x.transposed(0, 2, 1)
            let (w, v) = try eighAccelerated(s)
            XCTAssertLessThan(maxAbs(matmul(s, v) - v * w.expandedDimensions(axis: 1)), 1e-3)
            let (u, sv, vt) = try svdAccelerated(x)
            XCTAssertLessThan(maxAbs(matmul(u * sv.expandedDimensions(axis: 1), vt) - x), 1e-3)
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
