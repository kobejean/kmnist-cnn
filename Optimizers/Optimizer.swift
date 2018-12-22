//
//  Optimizer.swift
//  SwiftForTensorFlowTools
//
//  Created by Jean Flaherty on 12/10/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import TensorFlow

protocol Optimizer {
    associatedtype Parameters
    func optimize(_ θ: inout Parameters, _ dθ: Parameters)
}
