// QR, symmetric eigendecomposition and SVD on Apple GPUs, for batches of
// matrices held in [Float]: row-major, the matrices one after another. Each
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

func check(_ status: metal_linalg_status) throws {
    if status == METAL_LINALG_OK { return }
    let kind: MetalLinalgError.Kind
    switch status {
    case METAL_LINALG_INVALID_ARGUMENT: kind = .invalidArgument
    case METAL_LINALG_OUT_OF_MEMORY:    kind = .outOfMemory
    default:                            kind = .runtime
    }
    throw MetalLinalgError(kind: kind, message: String(cString: metal_linalg_last_error()))
}

func dimension(_ value: Int, _ name: String, _ who: String) throws -> UInt32 {
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

/// QR: A = Q R with K = min(rows, cols), for `batch` matrices of rows x cols.
/// Returns Q [batch, rows, K] with orthonormal columns and R [batch, K, cols]
/// upper triangular.
public func qrAccelerated(_ a: [Float], batch: Int = 1, rows: Int, cols: Int) throws -> (q: [Float], r: [Float]) {
    try checkCount(a.count, batch: batch, rows: rows, cols: cols, "qr")
    let (b, m, n) = (try dimension(batch, "batch", "qr"), try dimension(rows, "rows", "qr"), try dimension(cols, "cols", "qr"))
    let k = min(rows, cols)
    var r: [Float] = []
    let q = try a.withUnsafeBufferPointer { ap in
        try output(batch * rows * k) { qp in
            r = try output(batch * k * cols) { rp in
                try check(metal_linalg_qr(ap.baseAddress, b, m, n, qp, rp))
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

/// "cpu", "simd", "threadgroup", "block", "tridiag" or "ql".
public func eighBackend(n: Int, batch: Int = 1) -> String {
    String(cString: metal_linalg_eigh_backend(UInt32(clamping: n), UInt32(clamping: batch)))
}

/// The backend `eigvalshAccelerated` uses: as `eighBackend`, under the policy's
/// eigenvalues-alone boundary (`values_gpu_*`).
public func eigvalshBackend(n: Int, batch: Int = 1) -> String {
    String(cString: metal_linalg_eigvalsh_backend(UInt32(clamping: n), UInt32(clamping: batch)))
}

/// "cpu", "jacobi", "block_jacobi", "qr_jacobi", "qr_block_jacobi", "bidiag",
/// "golub_kahan" or "qr_golub_kahan".
public func svdBackend(rows: Int, cols: Int, batch: Int = 1) -> String {
    String(cString: metal_linalg_svd_backend(UInt32(clamping: rows), UInt32(clamping: cols), UInt32(clamping: batch)))
}

/// The backend `svdvalsAccelerated` uses: as `svdBackend`, with the policy's
/// `values_bidiag_min_k` for the bidiag backend.
public func svdvalsBackend(rows: Int, cols: Int, batch: Int = 1) -> String {
    String(cString: metal_linalg_svdvals_backend(UInt32(clamping: rows), UInt32(clamping: cols), UInt32(clamping: batch)))
}

/// The routing policies, field for field as in include/metal_linalg/core.h.
public typealias QrPolicy = metal_linalg_qr_policy
public typealias EighPolicy = metal_linalg_eigh_policy
public typealias SvdPolicy = metal_linalg_svd_policy

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

/// Where each policy came from: "tuned:<device>", "env:<variables>", "user"
/// or "default:untuned-device (<device>)".
public var qrPolicySource: String { String(cString: metal_linalg_qr_policy_source()) }
public var eighPolicySource: String { String(cString: metal_linalg_eigh_policy_source()) }
public var svdPolicySource: String { String(cString: metal_linalg_svd_policy_source()) }
