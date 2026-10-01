// The decompositions on MLXArray (mlx-swift), with the same names as the
// [Float] API in MetalLinalg. Batch dimensions are arbitrary; output is
// float32. Each call copies the input out of MLX and the results back in;
// the decompositions cost far more than the copies.
import MLX
@_exported import MetalLinalg

/// An MLXArray as the [Float] API takes it.
private struct Flat {
    let values: [Float]
    let batchShape: [Int]
    let batch: Int
    let rows: Int
    let cols: Int
}

private func flatten(_ a: MLXArray, _ who: String) throws -> Flat {
    let shape = a.shape
    guard shape.count >= 2 else {
        throw MetalLinalgError(kind: .invalidArgument, message: "[\(who)] Input must be at least a 2D matrix.")
    }
    let batchShape = Array(shape.dropLast(2))
    return Flat(values: a.asType(.float32).asArray(Float.self),
                batchShape: batchShape,
                batch: batchShape.reduce(1, *),
                rows: shape[shape.count - 2],
                cols: shape[shape.count - 1])
}

/// QR: A = Q R with K = min(M, N), for A [..., M, N]. Returns Q [..., M, K]
/// with orthonormal columns and R [..., K, N] upper triangular.
public func qrAccelerated(_ a: MLXArray) throws -> (q: MLXArray, r: MLXArray) {
    let x = try flatten(a, "qr")
    let k = min(x.rows, x.cols)
    let (q, r) = try qrAccelerated(x.values, batch: x.batch, rows: x.rows, cols: x.cols)
    return (MLXArray(q, x.batchShape + [x.rows, k]), MLXArray(r, x.batchShape + [k, x.cols]))
}

/// Symmetric eigendecomposition of A [..., N, N], reading one triangle.
/// Returns the eigenvalues [..., N] ascending and the eigenvectors as the
/// columns of [..., N, N].
public func eighAccelerated(_ a: MLXArray, uplo: Uplo = .lower) throws -> (eigenvalues: MLXArray, eigenvectors: MLXArray) {
    let x = try flatten(a, "eigh")
    let (w, v) = try eighAccelerated(x.values, batch: x.batch, n: x.cols, uplo: uplo)
    return (MLXArray(w, x.batchShape + [x.cols]), MLXArray(v, x.batchShape + [x.cols, x.cols]))
}

/// The eigenvalues alone, [..., N] ascending.
public func eigvalshAccelerated(_ a: MLXArray, uplo: Uplo = .lower) throws -> MLXArray {
    let x = try flatten(a, "eigvalsh")
    let w = try eigvalshAccelerated(x.values, batch: x.batch, n: x.cols, uplo: uplo)
    return MLXArray(w, x.batchShape + [x.cols])
}

/// Thin SVD of A [..., M, N], K = min(M, N): U [..., M, K], S [..., K]
/// descending, Vt [..., K, N].
public func svdAccelerated(_ a: MLXArray) throws -> (u: MLXArray, s: MLXArray, vt: MLXArray) {
    let x = try flatten(a, "svd")
    let k = min(x.rows, x.cols)
    let (u, s, vt) = try svdAccelerated(x.values, batch: x.batch, rows: x.rows, cols: x.cols)
    return (MLXArray(u, x.batchShape + [x.rows, k]),
            MLXArray(s, x.batchShape + [k]),
            MLXArray(vt, x.batchShape + [k, x.cols]))
}

/// The singular values alone, [..., K] descending.
public func svdvalsAccelerated(_ a: MLXArray) throws -> MLXArray {
    let x = try flatten(a, "svdvals")
    let s = try svdvalsAccelerated(x.values, batch: x.batch, rows: x.rows, cols: x.cols)
    return MLXArray(s, x.batchShape + [min(x.rows, x.cols)])
}
