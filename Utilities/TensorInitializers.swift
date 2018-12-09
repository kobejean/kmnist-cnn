//
//  File.swift
//  Iris
//
//  Created by Jean Flaherty on 12/1/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import TensorFlow

extension Tensor where Scalar : BinaryFloatingPoint & TensorFlowScalar, Scalar.RawSignificand : FixedWidthInteger {
    @inlinable @inline(__always)
    init(glorotUniform shape: TensorShape) {
        let minusOneToOne = 2 * Tensor(randomUniform: shape) - 1
        self = sqrt(Tensor(6 / Scalar(shape.contiguousSize))) * minusOneToOne
    }
}
