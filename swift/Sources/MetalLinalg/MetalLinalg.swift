// QR, symmetric eigendecomposition, SVD, Cholesky and LU (with solve and
// inverse) on Apple GPUs, for batches of matrices held in [Float]: row-major, the matrices one after another. Each
// call is routed to the fastest Metal kernel for its shape, or to LAPACK on
// the CPU, by a policy measured on the Mac it runs on. See docs/swift.md.
import CMetalLinalg

// MARK: - Errors

/// A shape a backend cannot take, a GPU failure, or a finite matrix that did
/// not converge. A NaN or an infinity in the input is not an error: that
/// matrix's results are NaN.
public struct MetalLinalgError: Error, CustomStringConvertible, Sendable {
    public enum Kind: Sendable { case invalidArgument, runtime, outOfMemory }
    public let kind: Kind
    public let message: String
    public var description: String { message }

    public init(kind: Kind, message: String) {
        self.kind = kind
        self.message = message
    }
}

package func check(_ status: metal_linalg_status) throws {
    if status == METAL_LINALG_OK { return }
    let kind: MetalLinalgError.Kind
    switch status {
    case METAL_LINALG_INVALID_ARGUMENT: kind = .invalidArgument
    case METAL_LINALG_OUT_OF_MEMORY:    kind = .outOfMemory
    default:                            kind = .runtime
    }
    throw MetalLinalgError(kind: kind, message: String(cString: metal_linalg_last_error()))
}

package func dimension(_ value: Int, _ name: String, _ who: String) throws -> UInt32 {
    guard let v = UInt32(exactly: value) else {
        throw MetalLinalgError(kind: .invalidArgument, message: "[\(who)] \(name) = \(value) is out of range.")
    }
    return v
}

func checkCount(_ count: Int, batch: Int, rows: Int, cols: Int, _ who: String) throws {
    let (n1, o1) = batch.multipliedReportingOverflow(by: rows)
    let (n, o2) = n1.multipliedReportingOverflow(by: cols)
    guard !o1, !o2, n == count else {
        throw MetalLinalgError(kind: .invalidArgument,
                               message: "[\(who)] \(count) values for \(batch) matrices of \(rows) x \(cols).")
    }
}

/// An array of `count` floats that `body` writes in full.
func output(_ count: Int, _ body: (UnsafeMutablePointer<Float>?) throws -> Void) rethrows -> [Float] {
    try [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
        try body(buffer.baseAddress)
        initialized = count
    }
}

// MARK: - Decompositions

/// Which factors QR returns, as numpy.linalg.qr's and torch.linalg.qr's modes.
public enum QrMode: Sendable {
    /// Q [rows, K] with orthonormal columns, R [K, cols].
    case reduced
    /// R [K, cols] alone; Q is empty and never formed (up to 2.8x faster).
    case r
    /// Q [rows, rows] square and orthogonal, R [rows, cols] with zero rows below K.
    case complete

    package var c: metal_linalg_qr_mode {
        switch self {
        case .reduced:  return METAL_LINALG_QR_REDUCED
        case .r:        return METAL_LINALG_QR_R
        case .complete: return METAL_LINALG_QR_COMPLETE
        }
    }

    /// Q's columns and R's rows for rows x cols matrices.
    package func shape(rows: Int, cols: Int) -> (qCols: Int, rRows: Int) {
        let k = min(rows, cols)
        switch self {
        case .reduced:  return (k, k)
        case .r:        return (0, k)
        case .complete: return (rows, rows)
        }
    }
}

/// QR: A = Q R with K = min(rows, cols), for `batch` matrices of rows x cols.
/// Returns Q [batch, rows, K] with orthonormal columns and R [batch, K, cols]
/// upper triangular; `mode` .r gives R alone (Q empty), .complete a square
/// Q [batch, rows, rows] and R [batch, rows, cols].
public func qrAccelerated(_ a: [Float], batch: Int = 1, rows: Int, cols: Int,
                          mode: QrMode = .reduced) throws -> (q: [Float], r: [Float]) {
    try checkCount(a.count, batch: batch, rows: rows, cols: cols, "qr")
    let (b, m, n) = (try dimension(batch, "batch", "qr"), try dimension(rows, "rows", "qr"), try dimension(cols, "cols", "qr"))
    let (qCols, rRows) = mode.shape(rows: rows, cols: cols)
    var r: [Float] = []
    let q = try a.withUnsafeBufferPointer { ap in
        try output(batch * rows * qCols) { qp in
            r = try output(batch * rRows * cols) { rp in
                try check(metal_linalg_qr_with_mode(ap.baseAddress, b, m, n, mode.c, qCols > 0 ? qp : nil, rp))
            }
        }
    }
    return (q, r)
}

/// Which triangle of a symmetric matrix is read; the other is ignored.
public enum Uplo: Sendable { case lower, upper }

/// Symmetric eigendecomposition A = V diag(w) V^T of `batch` matrices of
/// n x n. Returns the eigenvalues [batch, n] in ascending order and the
/// eigenvectors as the columns of [batch, n, n].
public func eighAccelerated(_ a: [Float], batch: Int = 1, n: Int, uplo: Uplo = .lower) throws
    -> (eigenvalues: [Float], eigenvectors: [Float])
{
    try checkCount(a.count, batch: batch, rows: n, cols: n, "eigh")
    let (b, nn) = (try dimension(batch, "batch", "eigh"), try dimension(n, "n", "eigh"))
    var v: [Float] = []
    let w = try a.withUnsafeBufferPointer { ap in
        try output(batch * n) { wp in
            v = try output(batch * n * n) { vp in
                try check(metal_linalg_eigh(ap.baseAddress, b, nn, uplo == .lower ? 1 : 0, wp, vp, nil))
            }
        }
    }
    return (w, v)
}

/// The eigenvalues alone, [batch, n] ascending; cheaper than with vectors.
public func eigvalshAccelerated(_ a: [Float], batch: Int = 1, n: Int, uplo: Uplo = .lower) throws -> [Float] {
    try checkCount(a.count, batch: batch, rows: n, cols: n, "eigvalsh")
    let (b, nn) = (try dimension(batch, "batch", "eigvalsh"), try dimension(n, "n", "eigvalsh"))
    return try a.withUnsafeBufferPointer { ap in
        try output(batch * n) { wp in
            try check(metal_linalg_eigh(ap.baseAddress, b, nn, uplo == .lower ? 1 : 0, wp, nil, nil))
        }
    }
}

/// Thin SVD A = U diag(S) Vt with K = min(rows, cols), for `batch` matrices
/// of rows x cols. Returns U [batch, rows, K], S [batch, K] descending and
/// Vt [batch, K, cols].
public func svdAccelerated(_ a: [Float], batch: Int = 1, rows: Int, cols: Int) throws
    -> (u: [Float], s: [Float], vt: [Float])
{
    try checkCount(a.count, batch: batch, rows: rows, cols: cols, "svd")
    let (b, m, n) = (try dimension(batch, "batch", "svd"), try dimension(rows, "rows", "svd"), try dimension(cols, "cols", "svd"))
    let k = min(rows, cols)
    var s: [Float] = []
    var vt: [Float] = []
    let u = try a.withUnsafeBufferPointer { ap in
        try output(batch * rows * k) { up in
            s = try output(batch * k) { sp in
                vt = try output(batch * k * cols) { vtp in
                    try check(metal_linalg_svd(ap.baseAddress, b, m, n, up, sp, vtp, nil))
                }
            }
        }
    }
    return (u, s, vt)
}

/// The singular values alone, [batch, K] descending; about half the work.
public func svdvalsAccelerated(_ a: [Float], batch: Int = 1, rows: Int, cols: Int) throws -> [Float] {
    try checkCount(a.count, batch: batch, rows: rows, cols: cols, "svdvals")
    let (b, m, n) = (try dimension(batch, "batch", "svdvals"), try dimension(rows, "rows", "svdvals"), try dimension(cols, "cols", "svdvals"))
    return try a.withUnsafeBufferPointer { ap in
        try output(batch * min(rows, cols)) { sp in
            try check(metal_linalg_svd(ap.baseAddress, b, m, n, nil, sp, nil, nil))
        }
    }
}

/// Cholesky A = L L^T of `batch` symmetric positive definite matrices of
/// n x n, reading the lower triangle (the upper with `upper`, then returning
/// U = L^T). Returns L [batch, n, n] with zeros in the other triangle and
/// `info` [batch]: 0, or k where the leading minor of order k is not positive
/// definite (as LAPACK's spotrf), and then that matrix's L is all NaN.
public func choleskyAccelerated(_ a: [Float], batch: Int = 1, n: Int, upper: Bool = false) throws
    -> (l: [Float], info: [UInt32])
{
    try checkCount(a.count, batch: batch, rows: n, cols: n, "cholesky")
    let (b, nn) = (try dimension(batch, "batch", "cholesky"), try dimension(n, "n", "cholesky"))
    var info = [UInt32](repeating: 0, count: batch)
    let l = try a.withUnsafeBufferPointer { ap in
        try info.withUnsafeMutableBufferPointer { ip in
            try output(batch * n * n) { lp in
                try check(metal_linalg_cholesky(ap.baseAddress, b, nn, upper ? 1 : 0, lp, ip.baseAddress))
            }
        }
    }
    return (l, info)
}

/// LU with partial pivoting, P A = L U, of `batch` n x n matrices (since
/// 2.18.0): the packed factors [batch, n, n] (U on and above the diagonal, L
/// below it, its unit diagonal implied), the pivots [batch, n] (0-based: row i
/// was swapped with row pivots[i], in order) and `info` [batch] (0, or k where
/// U's k-th diagonal entry is exactly zero).
public func luFactorAccelerated(_ a: [Float], batch: Int = 1, n: Int) throws
    -> (lu: [Float], pivots: [UInt32], info: [UInt32])
{
    try checkCount(a.count, batch: batch, rows: n, cols: n, "lu_factor")
    let (b, nn) = (try dimension(batch, "batch", "lu_factor"), try dimension(n, "n", "lu_factor"))
    var pivots = [UInt32](repeating: 0, count: batch * n)
    var info = [UInt32](repeating: 0, count: batch)
    let lu = try a.withUnsafeBufferPointer { ap in
        try pivots.withUnsafeMutableBufferPointer { pp in
            try info.withUnsafeMutableBufferPointer { ip in
                try output(batch * n * n) { lp in
                    try check(metal_linalg_lu_factor(ap.baseAddress, b, nn, lp, pp.baseAddress, ip.baseAddress))
                }
            }
        }
    }
    return (lu, pivots, info)
}

/// X with A X = B for `batch` n x n matrices and B [batch, n, nrhs] (X the
/// same); `info` as luFactorAccelerated's, a singular matrix's X all NaN.
public func solveAccelerated(_ a: [Float], _ b: [Float], batch: Int = 1, n: Int, nrhs: Int = 1) throws
    -> (x: [Float], info: [UInt32])
{
    try checkCount(a.count, batch: batch, rows: n, cols: n, "solve")
    try checkCount(b.count, batch: batch, rows: n, cols: nrhs, "solve")
    let (bt, nn, k) = (try dimension(batch, "batch", "solve"), try dimension(n, "n", "solve"),
                       try dimension(nrhs, "nrhs", "solve"))
    var info = [UInt32](repeating: 0, count: batch)
    let x = try a.withUnsafeBufferPointer { ap in
        try b.withUnsafeBufferPointer { bp in
            try info.withUnsafeMutableBufferPointer { ip in
                try output(batch * n * nrhs) { xp in
                    try check(metal_linalg_solve(ap.baseAddress, bt, nn, bp.baseAddress, k, xp, ip.baseAddress))
                }
            }
        }
    }
    return (x, info)
}

/// A^-1 [batch, n, n]; `info` as luFactorAccelerated's, a singular matrix's
/// inverse all NaN.
public func invAccelerated(_ a: [Float], batch: Int = 1, n: Int) throws -> (x: [Float], info: [UInt32]) {
    try checkCount(a.count, batch: batch, rows: n, cols: n, "inv")
    let (b, nn) = (try dimension(batch, "batch", "inv"), try dimension(n, "n", "inv"))
    var info = [UInt32](repeating: 0, count: batch)
    let x = try a.withUnsafeBufferPointer { ap in
        try info.withUnsafeMutableBufferPointer { ip in
            try output(batch * n * n) { xp in
                try check(metal_linalg_inv(ap.baseAddress, b, nn, xp, ip.baseAddress))
            }
        }
    }
    return (x, info)
}

/// X with A X = B for `batch` triangular n x n matrices and B [batch, n,
/// nrhs] (X the same), reading A's lower triangle (the upper with `upper`;
/// with `unitDiagonal` its diagonal taken as ones). Since 2.18.0.
public func solveTriangularAccelerated(_ a: [Float], _ b: [Float], batch: Int = 1, n: Int, nrhs: Int = 1,
                                       upper: Bool = false, unitDiagonal: Bool = false) throws -> [Float] {
    try checkCount(a.count, batch: batch, rows: n, cols: n, "solve_triangular")
    try checkCount(b.count, batch: batch, rows: n, cols: nrhs, "solve_triangular")
    let (bt, nn, k) = (try dimension(batch, "batch", "solve_triangular"), try dimension(n, "n", "solve_triangular"),
                       try dimension(nrhs, "nrhs", "solve_triangular"))
    return try a.withUnsafeBufferPointer { ap in
        try b.withUnsafeBufferPointer { bp in
            try output(batch * n * nrhs) { xp in
                try check(metal_linalg_solve_triangular(ap.baseAddress, bt, nn, bp.baseAddress, k, upper ? 1 : 0,
                                                        unitDiagonal ? 1 : 0, xp))
            }
        }
    }
}

// MARK: - Device and routing

/// The default Metal device, e.g. "Apple M5 Pro"; empty if there is none.
public var deviceName: String { String(cString: metal_linalg_device_name()) }

/// Its GPU core count; 0 if it could not be read.
public var gpuCoreCount: Int { Int(metal_linalg_gpu_core_count()) }

/// CPU threads the CPU paths spread a batch over: every core by default. Cap it
/// for a program that runs several solves at once on threads of its own;
/// setting 0 restores every core. `METAL_LINALG_CPU_THREADS` sets it from the
/// environment.
public var cpuThreads: Int {
    get { Int(metal_linalg_cpu_threads()) }
    set { metal_linalg_set_cpu_threads(UInt32(clamping: newValue)) }
}

/// The backend a call of that shape uses: "cpu", "unblocked" or "streaming_reduced".
public func qrBackend(rows: Int, cols: Int, batch: Int = 1) -> String {
    String(cString: metal_linalg_qr_backend(UInt32(clamping: rows), UInt32(clamping: cols), UInt32(clamping: batch)))
}

/// "cpu", "simd", "threadgroup", "block", "tridiag", "ql", "band" (the
/// two-stage reduction, from the policy's `band_min_n`) or "tridiag_batch" (a
/// batch of mid-size matrices at once, inside the policy's `tridiag_batch_*`
/// window).
public func eighBackend(n: Int, batch: Int = 1) -> String {
    String(cString: metal_linalg_eigh_backend(UInt32(clamping: n), UInt32(clamping: batch)))
}

/// The backend `eigvalshAccelerated` uses: as `eighBackend`, under the policy's
/// eigenvalues-alone boundary (`values_gpu_*`), and "band" (the two-stage
/// reduction) from `values_band_min_n`.
public func eigvalshBackend(n: Int, batch: Int = 1) -> String {
    String(cString: metal_linalg_eigvalsh_backend(UInt32(clamping: n), UInt32(clamping: batch)))
}

/// "cpu", "jacobi", "block_jacobi", "qr_jacobi", "qr_block_jacobi", "bidiag",
/// "band" (the two-stage reduction, from the policy's `band_min_k`),
/// "golub_kahan", "qr_golub_kahan" or "bidiag_batch" (a batch of mid-size
/// matrices at once, inside the policy's `bidiag_batch_*` window).
public func svdBackend(rows: Int, cols: Int, batch: Int = 1) -> String {
    String(cString: metal_linalg_svd_backend(UInt32(clamping: rows), UInt32(clamping: cols), UInt32(clamping: batch)))
}

/// The backend `svdvalsAccelerated` uses: as `svdBackend`, with the policy's
/// `values_bidiag_min_k` for the bidiag backend, and "band" (the two-stage
/// reduction) from `values_band_min_k`.
public func svdvalsBackend(rows: Int, cols: Int, batch: Int = 1) -> String {
    String(cString: metal_linalg_svdvals_backend(UInt32(clamping: rows), UInt32(clamping: cols), UInt32(clamping: batch)))
}

/// "cpu", "simd" (up to 32 x 32), "threadgroup" or "blocked" (the
/// large-matrix path).
public func choleskyBackend(n: Int, batch: Int = 1) -> String {
    String(cString: metal_linalg_cholesky_backend(UInt32(clamping: n), UInt32(clamping: batch)))
}

/// The routing policies, field for field as in include/metal_linalg/core.h.
public typealias QrPolicy = metal_linalg_qr_policy
public typealias EighPolicy = metal_linalg_eigh_policy
public typealias SvdPolicy = metal_linalg_svd_policy
public typealias CholeskyPolicy = metal_linalg_cholesky_policy
public typealias LuPolicy = metal_linalg_lu_policy
public typealias TrsmPolicy = metal_linalg_trsm_policy

/// "cpu" or "blocked" (the GPU path), for solveTriangularAccelerated.
public func trsmBackend(n: Int, nrhs: Int = 1, batch: Int = 1) -> String {
    String(cString: metal_linalg_trsm_backend(UInt32(clamping: n), UInt32(clamping: nrhs), UInt32(clamping: batch)))
}

/// "cpu" or "blocked" (the GPU path), for lu_factor, solve and inv alike.
public func luBackend(n: Int, batch: Int = 1) -> String {
    String(cString: metal_linalg_lu_backend(UInt32(clamping: n), UInt32(clamping: batch)))
}

/// `gpu_max_n` / `gpu_max_k` value meaning no cap.
public let policyNoLimit: UInt32 = 0xFFFF_FFFF

/// The policy in effect. Setting one replaces the tuned table and the
/// environment for the rest of the process.
public var qrPolicy: QrPolicy {
    get { metal_linalg_qr_policy_get() }
    set { withUnsafePointer(to: newValue) { metal_linalg_qr_policy_set($0) } }
}

public var eighPolicy: EighPolicy {
    get { metal_linalg_eigh_policy_get() }
    set { withUnsafePointer(to: newValue) { metal_linalg_eigh_policy_set($0) } }
}

public var svdPolicy: SvdPolicy {
    get { metal_linalg_svd_policy_get() }
    set { withUnsafePointer(to: newValue) { metal_linalg_svd_policy_set($0) } }
}

public var choleskyPolicy: CholeskyPolicy {
    get { metal_linalg_cholesky_policy_get() }
    set { withUnsafePointer(to: newValue) { metal_linalg_cholesky_policy_set($0) } }
}

public var luPolicy: LuPolicy {
    get { metal_linalg_lu_policy_get() }
    set { withUnsafePointer(to: newValue) { metal_linalg_lu_policy_set($0) } }
}

public var trsmPolicy: TrsmPolicy {
    get { metal_linalg_trsm_policy_get() }
    set { withUnsafePointer(to: newValue) { metal_linalg_trsm_policy_set($0) } }
}

/// Where each policy came from: "tuned:<device>", "env:<variables>", "user"
/// or "default:untuned-device (<device>)".
public var qrPolicySource: String { String(cString: metal_linalg_qr_policy_source()) }
public var eighPolicySource: String { String(cString: metal_linalg_eigh_policy_source()) }
public var svdPolicySource: String { String(cString: metal_linalg_svd_policy_source()) }
public var choleskyPolicySource: String { String(cString: metal_linalg_cholesky_policy_source()) }
public var luPolicySource: String { String(cString: metal_linalg_lu_policy_source()) }
public var trsmPolicySource: String { String(cString: metal_linalg_trsm_policy_source()) }
