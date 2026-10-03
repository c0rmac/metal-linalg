// Tests of the Swift API: every function reaches the library with the right
// shapes, values and errors. The kernels are tested in depth by the C++ suites.
import XCTest
@testable import MetalLinalg

final class MetalLinalgTests: XCTestCase {
    // Deterministic pseudo-random values in [-1, 1).
    func values(_ count: Int, seed: UInt64) -> [Float] {
        var state = seed &* 0x9E37_79B9_7F4A_7C15 | 1
        return (0..<count).map { _ in
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            return Float(Double(state >> 11) / Double(1 << 53) * 2 - 1)
        }
    }

    // C = A B for row-major A (m x k), B (k x n), in double.
    func multiply(_ a: ArraySlice<Float>, _ b: ArraySlice<Float>, _ m: Int, _ k: Int, _ n: Int) -> [Double] {
        let a = Array(a), b = Array(b)
        var c = [Double](repeating: 0, count: m * n)
        for i in 0..<m { for j in 0..<n { for t in 0..<k { c[i * n + j] += Double(a[i * k + t]) * Double(b[t * n + j]) } } }
        return c
    }

    func testQR() throws {
        let (batch, m, n) = (5, 30, 12)
        let a = values(batch * m * n, seed: 1)
        let (q, r) = try qrAccelerated(a, batch: batch, rows: m, cols: n)
        XCTAssertEqual(q.count, batch * m * n)
        XCTAssertEqual(r.count, batch * n * n)
        for b in 0..<batch {
            let qr = multiply(q[(b * m * n)..<((b + 1) * m * n)], r[(b * n * n)..<((b + 1) * n * n)], m, n, n)
            for i in 0..<(m * n) { XCTAssertEqual(qr[i], Double(a[b * m * n + i]), accuracy: 1e-5) }
            for i in 0..<n { for j in 0..<i { XCTAssertEqual(r[b * n * n + i * n + j], 0) } }
        }
    }

    func testEigh() throws {
        let (batch, n) = (64, 8)
        var a = values(batch * n * n, seed: 2)
        for b in 0..<batch { for i in 0..<n { for j in 0..<i { a[b * n * n + i * n + j] = a[b * n * n + j * n + i] } } }
        let (w, v) = try eighAccelerated(a, batch: batch, n: n)
        for b in 0..<batch {
            let av = multiply(a[(b * n * n)..<((b + 1) * n * n)], v[(b * n * n)..<((b + 1) * n * n)], n, n, n)
            for i in 0..<n {
                for j in 0..<n { XCTAssertEqual(av[i * n + j], Double(v[b * n * n + i * n + j] * w[b * n + j]), accuracy: 1e-4) }
                if i > 0 { XCTAssertLessThanOrEqual(w[b * n + i - 1], w[b * n + i]) }
            }
        }
        let wOnly = try eigvalshAccelerated(a, batch: batch, n: n, uplo: .upper)
        for i in 0..<w.count { XCTAssertEqual(wOnly[i], w[i], accuracy: 1e-4) }
    }

    // The bidiag SVD backend, forced by the policy: square, tall (QR first)
    // and wide, so its shader library and the QR it calls are exercised.
    func testSVDBidiag() throws {
        let measured = svdPolicy
        defer { svdPolicy = measured }
        var p = measured
        p.gpu_max_k = 0
        p.bidiag_min_k = 1
        p.values_bidiag_min_k = 1
        svdPolicy = p
        for (rows, cols) in [(70, 70), (300, 80), (60, 150)] {
            XCTAssertEqual(svdBackend(rows: rows, cols: cols), "bidiag")
            let a = values(rows * cols, seed: UInt64(rows + cols))
            let (u, s, vt) = try svdAccelerated(a, batch: 1, rows: rows, cols: cols)
            let k = min(rows, cols)
            var maxErr = 0.0
            for i in 0..<rows {
                for j in 0..<cols {
                    var sum = 0.0
                    for t in 0..<k { sum += Double(u[i * k + t]) * Double(s[t]) * Double(vt[t * cols + j]) }
                    maxErr = max(maxErr, abs(sum - Double(a[i * cols + j])))
                }
            }
            XCTAssertLessThan(maxErr, 1e-3)
            for t in 1..<k { XCTAssertGreaterThanOrEqual(s[t - 1], s[t]) }
            let sOnly = try svdvalsAccelerated(a, batch: 1, rows: rows, cols: cols)
            for t in 0..<k { XCTAssertEqual(sOnly[t], s[t], accuracy: 1e-3) }
        }
    }

    // The tridiag backend, forced by the policy, on sizes either side of a
    // GPU panel (32 columns): it loads its own shader library and runs MPS.
    // Eigenvalues alone on the CPU, where from N = 128 LAPACK's two-stage
    // driver is used when it can be trusted: macOS 14's gave values off by
    // several percent, so it must agree with the eigenvalues of eigh here.
    func testCPUEigenvaluesAlone() throws {
        let measured = eighPolicy
        defer { eighPolicy = measured }
        var p = measured
        p.gpu_max_n = 0; p.gpu_min_batch_times_n = 0
        p.values_gpu_max_n = 0; p.values_gpu_min_batch_times_n = 0; p.values_gpu_min_batch = 1
        p.tridiag_min_n = 0; p.values_tridiag_min_n = 0
        eighPolicy = p
        for n in [127, 128, 200, 300] {
            XCTAssertEqual(eigvalshBackend(n: n), "cpu")
            var a = values(n * n, seed: UInt64(40 + n))
            for i in 0..<n { for j in 0..<i { a[i * n + j] = a[j * n + i] } }
            let w = try eighAccelerated(a, n: n).0
            let scale = w.map { abs($0) }.max()!
            for uplo in [Uplo.lower, .upper] {
                let wOnly = try eigvalshAccelerated(a, n: n, uplo: uplo)
                for i in 0..<n { XCTAssertEqual(wOnly[i], w[i], accuracy: 1e-5 * scale, "N=\(n) \(uplo) i=\(i)") }
            }
        }
    }

    func testEighTridiag() throws {
        let measured = eighPolicy
        defer { eighPolicy = measured }
        var p = measured
        p.gpu_max_n = 0              // never the Jacobi backends
        p.tridiag_min_n = 1          // so every eigh is tridiag
        p.values_tridiag_min_n = 1
        eighPolicy = p
        for n in [3, 70, 150] {
            XCTAssertEqual(eighBackend(n: n, batch: 2), "tridiag")
            let batch = 2
            var a = values(batch * n * n, seed: UInt64(10 + n))
            for b in 0..<batch { for i in 0..<n { for j in 0..<i { a[b * n * n + i * n + j] = a[b * n * n + j * n + i] } } }
            let (w, v) = try eighAccelerated(a, batch: batch, n: n)
            for b in 0..<batch {
                let av = multiply(a[(b * n * n)..<((b + 1) * n * n)], v[(b * n * n)..<((b + 1) * n * n)], n, n, n)
                for i in 0..<n {
                    for j in 0..<n { XCTAssertEqual(av[i * n + j], Double(v[b * n * n + i * n + j] * w[b * n + j]), accuracy: 1e-3) }
                    if i > 0 { XCTAssertLessThanOrEqual(w[b * n + i - 1], w[b * n + i]) }
                }
            }
            let wOnly = try eigvalshAccelerated(a, batch: batch, n: n)
            for i in 0..<w.count { XCTAssertEqual(wOnly[i], w[i], accuracy: 1e-3) }
        }
    }

    func testSVDTallAndWide() throws {
        for (m, n) in [(20, 7), (7, 20)] {
            let batch = 4, k = min(m, n)
            let a = values(batch * m * n, seed: UInt64(m * 100 + n))
            let (u, s, vt) = try svdAccelerated(a, batch: batch, rows: m, cols: n)
            XCTAssertEqual([u.count, s.count, vt.count], [batch * m * k, batch * k, batch * k * n])
            for b in 0..<batch {
                var us = Array(u[(b * m * k)..<((b + 1) * m * k)])
                for i in 0..<m { for j in 0..<k { us[i * k + j] *= s[b * k + j] } }
                let usv = multiply(us[...], vt[(b * k * n)..<((b + 1) * k * n)], m, k, n)
                for i in 0..<(m * n) { XCTAssertEqual(usv[i], Double(a[b * m * n + i]), accuracy: 1e-5) }
                for j in 1..<k { XCTAssertGreaterThanOrEqual(s[b * k + j - 1], s[b * k + j]) }
            }
            let sOnly = try svdvalsAccelerated(a, batch: batch, rows: m, cols: n)
            for i in 0..<s.count { XCTAssertEqual(sOnly[i], s[i], accuracy: 1e-5) }
        }
    }

    func testNaNStaysInItsMatrix() throws {
        var a: [Float] = [2, 1, 1, 2,  .nan, 0, 0, 1,  3, 0, 0, 4]
        let w = try eigvalshAccelerated(a, batch: 3, n: 2)
        XCTAssertFalse(w[0].isNaN || w[1].isNaN || w[4].isNaN || w[5].isNaN)
        XCTAssertTrue(w[2].isNaN && w[3].isNaN)
        a[4] = 0
        XCTAssertFalse(try eigvalshAccelerated(a, batch: 3, n: 2).contains { $0.isNaN })
    }

    func testErrors() {
        XCTAssertThrowsError(try qrAccelerated([1, 2, 3], rows: 2, cols: 2)) { error in
            XCTAssertEqual((error as? MetalLinalgError)?.kind, .invalidArgument)
        }
        XCTAssertThrowsError(try svdAccelerated([1, 2], batch: -1, rows: 1, cols: 2))
    }

    func testRoutingAndPolicies() {
        XCTAssertFalse(deviceName.isEmpty)
        XCTAssertGreaterThanOrEqual(gpuCoreCount, 0)   // 0 means unknown, as on a virtual GPU
        XCTAssertTrue(["unblocked", "streaming_reduced"].contains(qrBackend(rows: 64, cols: 64, batch: 100)))
        let measured = eighPolicy
        defer { eighPolicy = measured }
        var p = measured
        p.gpu_max_n = 0
        eighPolicy = p
        XCTAssertEqual(eighBackend(n: 8, batch: 4096), "cpu")
        XCTAssertEqual(eighPolicySource, "user")
        // Eigenvalues alone: unset (values_gpu_min_batch = 0) follows eigh; set, it decides apart.
        p.values_gpu_min_batch = 0
        eighPolicy = p
        XCTAssertEqual(eigvalshBackend(n: 8, batch: 4096), "cpu")
        p.values_gpu_max_n = 64
        p.values_gpu_min_batch_times_n = 0
        p.values_gpu_min_batch = 1
        eighPolicy = p
        XCTAssertNotEqual(eigvalshBackend(n: 8, batch: 4096), "cpu")
        XCTAssertEqual(eighBackend(n: 8, batch: 4096), "cpu")
    }
}
