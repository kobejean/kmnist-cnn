//
//  MathSymbols.swift
//  SwiftForTensorFlowTools
//
//  Created by Jean Flaherty on 12/8/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import TensorFlow

// Math Symbols

@inlinable @inline(__always)
func μ<T: Numeric>(_ x: Tensor<T>) -> T {
    return x.mean()
}

@inlinable @inline(__always)
func μ<T: Numeric>(_ x: Tensor<T>, alongAxes: Int32...) -> Tensor<T> {
    let indices = Tensor(alongAxes)
    return Raw.mean(x, reductionIndices: indices, keepDims: true)
}

@inlinable @inline(__always)
func μ<T: Numeric>(_ x: Tensor<T>, reducingAxes: Int32...) -> Tensor<T> {
    let indices = Tensor(reducingAxes)
    return Raw.mean(x, reductionIndices: indices, keepDims: false)
}

@inlinable @inline(__always)
func e_i<T: Numeric>(_ x: Tensor<Int32>, _ n: Int32) -> Tensor<T> {
    return Tensor<T>(oneHotAtIndices: x, depth: n)
}

postfix operator ⊺
extension Tensor {
    @inlinable @inline(__always)
    static postfix func ⊺ (lhs: Tensor) -> Tensor {
        return lhs.transposed()
    }
}

prefix operator ∏
prefix operator ∑
extension Tensor where Scalar : Numeric {
    @inlinable @inline(__always)
    static prefix func ∏ (_ x: Tensor) -> Tensor {
        return x.product(squeezingAxes: -1)
    }
    @inlinable @inline(__always)
    static prefix func ∑ (_ x: Tensor) -> Tensor {
        return x.sum(squeezingAxes: -1)
    }
}

prefix operator √
extension Tensor where Scalar: BinaryFloatingPoint {
    @inlinable @inline(__always)
    static prefix func √ (rhs: Tensor) -> Tensor {
        return sqrt(rhs)
    }
}
