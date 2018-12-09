//
//  MathSymbols.swift
//  Iris
//
//  Created by Jean Flaherty on 12/8/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import TensorFlow

// Math Symbols
@inlinable @inline(__always) func Σ(_ x: Tensor<Float>, _ N: Int32 = -1) -> Tensor<Float> { return x.sum(squeezingAxes: N) }
@inlinable @inline(__always) func μ(_ x: Tensor<Float>) -> Float { return x.mean() }
@inlinable @inline(__always) func e_i(_ x: Tensor<Int32>, _ n: Int32) -> Tensor<Float> { return oneHot(indices: x, depth: n, onValue: 1, offValue: 0) }
postfix operator ⊺
extension Tensor where Scalar == Float {
    @inlinable @inline(__always) static postfix func ⊺ (lhs: Tensor) -> Tensor { return lhs.transposed() }
}
