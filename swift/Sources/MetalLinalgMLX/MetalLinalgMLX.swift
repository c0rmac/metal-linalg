// The decompositions on MLXArray (mlx-swift), with the same names as the
// [Float] API in MetalLinalg. Batch dimensions are arbitrary; output is
// float32. The input is read in place once MLX has evaluated it (a strided
// view is copied to contiguous memory first), and the results are written
// into page-aligned memory that each output MLXArray then owns: MLX wraps it
// as a Metal buffer without a copy where Metal accepts it, and copies it once
// where it does not.
import CMetalLinalg
import Darwin
import MLX
@_exported import MetalLinalg

/// An MLXArray as the C API takes it.
private struct Matrices {
    let array: MLXArray     // float32; owns the memory the C API reads
    let batchShape: [Int]
    let batch: UInt32
    let rows: UInt32
    let cols: UInt32
    let count: Int

    init(_ a: MLXArray, _ who: String) throws {
        let shape = a.shape
        guard shape.count >= 2 else {
            throw MetalLinalgError(kind: .invalidArgument, message: "[\(who)] Input must be at least a 2D matrix.")
        }
        // Real input only: casting complex to float32 would keep the real parts.
        guard a.dtype != .complex64 else {
            throw MetalLinalgError(kind: .invalidArgument,
                                   message: "[\(who)] Complex input is not supported: the decompositions are real (float32).")
        }
        batchShape = Array(shape.dropLast(2))
        let b = batchShape.reduce(1, *)
        batch = try dimension(b, "batch", who)
        rows = try dimension(shape[shape.count - 2], "rows", who)
        cols = try dimension(shape[shape.count - 1], "cols", who)
        count = b * shape[shape.count - 2] * shape[shape.count - 1]
        array = a.asType(.float32)
    }

    /// New float32 MLXArrays of `shapes`, written by `call` from the input's
    /// memory and the outputs' (one pointer each, in order).
    func run(_ shapes: [[Int]],
             _ call: (UnsafePointer<Float>?, [UnsafeMutablePointer<Float>]) -> metal_linalg_status) throws -> [MLXArray] {
        if count == 0 {   // nothing to read, and every output is empty too
            return shapes.map { MLXArray.zeros($0, type: Float.self) }
        }
        var outputs: [UnsafeMutablePointer<Float>] = []
        defer { outputs.forEach { free($0) } }   // whatever has not become an array
        for shape in shapes {
            guard let p = pageAligned(shape.reduce(1, *)) else {
                throw MetalLinalgError(kind: .outOfMemory, message: "out of memory")
            }
            outputs.append(p)
        }
        // Evaluates the input; its own memory, unless it is a strided view.
        let input = array.asData(access: .noCopyIfContiguous)
        let status = withExtendedLifetime(array) {
            input.data.withUnsafeBytes { call($0.bindMemory(to: Float.self).baseAddress, outputs) }
        }
        try check(status)
        let arrays = zip(outputs, shapes).map { p, shape in
            MLXArray(rawPointer: UnsafeMutableRawPointer(p), shape, dtype: .float32) { free(p) }
        }
        outputs = []
        return arrays
    }
}

/// `count` floats of memory starting on a page boundary and filling whole
/// pages, which Metal wraps as a buffer in place; nil if it cannot be had.
private func pageAligned(_ count: Int) -> UnsafeMutablePointer<Float>? {
    let page = Int(getpagesize())
    let bytes = (count * MemoryLayout<Float>.stride + page - 1) / page * page
    var p: UnsafeMutableRawPointer?
    guard posix_memalign(&p, page, bytes) == 0, let p else { return nil }
    return p.bindMemory(to: Float.self, capacity: count)
}

/// QR: A = Q R with K = min(M, N), for A [..., M, N]. Returns Q [..., M, K]
/// with orthonormal columns and R [..., K, N] upper triangular; `mode` .r
/// gives R alone (Q an empty array, never formed), .complete a square
/// Q [..., M, M] and R [..., M, N] with zero rows below K.
public func qrAccelerated(_ a: MLXArray, mode: QrMode = .reduced) throws -> (q: MLXArray, r: MLXArray) {
    let x = try Matrices(a, "qr")
    let m = Int(x.rows), n = Int(x.cols)
    let (qCols, rRows) = mode.shape(rows: m, cols: n)
    let rShape = x.batchShape + [rRows, n]
    if mode == .r {
        let r = try x.run([rShape]) { a, o in
            metal_linalg_qr_with_mode(a, x.batch, x.rows, x.cols, METAL_LINALG_QR_R, nil, o[0])
        }[0]
        return (MLXArray.zeros([0], type: Float.self), r)
    }
    let qShape = x.batchShape + [m, qCols]
    if mode == .complete && n == 0 {   // nothing to factor: Q is the identity
        return (broadcast(MLXArray.identity(m, type: Float.self), to: qShape), MLXArray.zeros(rShape, type: Float.self))
    }
    let out = try x.run([qShape, rShape]) { a, o in
        metal_linalg_qr_with_mode(a, x.batch, x.rows, x.cols, mode.c, o[0], o[1])
    }
    return (out[0], out[1])
}

/// Symmetric eigendecomposition of A [..., N, N], reading one triangle.
/// Returns the eigenvalues [..., N] ascending and the eigenvectors as the
/// columns of [..., N, N].
public func eighAccelerated(_ a: MLXArray, uplo: Uplo = .lower) throws -> (eigenvalues: MLXArray, eigenvectors: MLXArray) {
    let x = try Matrices(a, "eigh")
    try square(x, "eigh")
    let n = Int(x.cols)
    let out = try x.run([x.batchShape + [n], x.batchShape + [n, n]]) { a, o in
        metal_linalg_eigh(a, x.batch, x.cols, uplo == .lower ? 1 : 0, o[0], o[1], nil)
    }
    return (out[0], out[1])
}

/// The eigenvalues alone, [..., N] ascending.
public func eigvalshAccelerated(_ a: MLXArray, uplo: Uplo = .lower) throws -> MLXArray {
    let x = try Matrices(a, "eigvalsh")
    try square(x, "eigvalsh")
    return try x.run([x.batchShape + [Int(x.cols)]]) { a, o in
        metal_linalg_eigh(a, x.batch, x.cols, uplo == .lower ? 1 : 0, o[0], nil, nil)
    }[0]
}

/// Thin SVD of A [..., M, N], K = min(M, N): U [..., M, K], S [..., K]
/// descending, Vt [..., K, N].
public func svdAccelerated(_ a: MLXArray) throws -> (u: MLXArray, s: MLXArray, vt: MLXArray) {
    let x = try Matrices(a, "svd")
    let m = Int(x.rows), n = Int(x.cols), k = min(m, n)
    let out = try x.run([x.batchShape + [m, k], x.batchShape + [k], x.batchShape + [k, n]]) { a, o in
        metal_linalg_svd(a, x.batch, x.rows, x.cols, o[0], o[1], o[2], nil)
    }
    return (out[0], out[1], out[2])
}

/// The singular values alone, [..., K] descending.
public func svdvalsAccelerated(_ a: MLXArray) throws -> MLXArray {
    let x = try Matrices(a, "svdvals")
    return try x.run([x.batchShape + [Int(min(x.rows, x.cols))]]) { a, o in
        metal_linalg_svd(a, x.batch, x.rows, x.cols, nil, o[0], nil, nil)
    }[0]
}

private func square(_ x: Matrices, _ who: String) throws {
    guard x.rows == x.cols else {
        throw MetalLinalgError(kind: .invalidArgument, message: "[\(who)] Input matrices must be square.")
    }
}
