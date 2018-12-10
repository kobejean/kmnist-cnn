//
//  AdamOptimizer.swift
//  kmnist
//
//  Created by Jean Flaherty on 12/9/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import TensorFlow

@usableFromInline
class AdamOptimizer<Parameters: ParameterGroup> where Parameters.Parameter == Tensor<Float> {
    typealias Scalar = Float
    let α: Scalar
    let βm: Scalar
    let βv: Scalar
    let ϵ: Scalar
    var m: Parameters
    var v: Parameters
    
    init(_ θ: Parameters, _ α: Scalar = 0.001, _ βm: Scalar = 0.9, _ βv: Scalar = 0.999, _ ϵ: Scalar = 1e-08) {
        self.α = α
        self.βm = βm
        self.βv = βv
        self.ϵ = ϵ
        var zero = θ
        zero.update(withGradients: θ) { (m_k, θ_k) in m_k = 0 * θ_k }
        self.m = zero
        self.v = zero
    }
    
    @usableFromInline
    func optimize(_ θ: inout Parameters, _ dθ: Parameters) {
        m.update(withGradients: dθ) { (m_k, dθ_k) in m_k = βm * m_k + (1 - βm) * dθ_k }
        v.update(withGradients: dθ) { (v_k, dθ_k) in v_k = βv * v_k + (1 - βv) * dθ_k * dθ_k }
        var Δθ = m
        Δθ.update(withGradients: v) { (Δθ_k, v_k) in
            let m_hat = Δθ_k / (1 - βm)
            let v_hat = v_k / (1 - βv)
            Δθ_k = α * m_hat / (√v_hat + ϵ)
        }
        θ.update(withGradients: Δθ) { (θ_k, Δθ_k) in θ_k -= Δθ_k }
    }
}
