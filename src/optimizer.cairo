use dict::Felt252DictTrait;
use orion::operators::tensor::core::Tensor;
use orion::numbers::signed_integer::i32::i32;
use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::numbers::fixed_point::implementations::impl_8x23::FP8x23Impl;
use orion::numbers::fixed_point::implementations::impl_8x23::FP8x23Sub;
use orion::numbers::fixed_point::implementations::impl_8x23::FP8x23Mul;
use orion::numbers::fixed_point::implementations::impl_8x23::FP8x23Div;
use optimizer_utils::{exponential_weights, weighted_covariance, rolling_covariance, diagonalize};

#[starknet::contract]
mod BlackLittermanOptimizer {
    use starknet::ContractAddress;
    use starknet::get_caller_address;

    struct Storage {
        contract_owner: ContractAddress,
        tickers: Array::<felt252>, // total list of tickers, fixed on L1
        views_tickers: Array::<felt252>, //subset of tickers that have views associated with them, externally set from L1
        views: LegacyMap::<felt252, (u32, felt252)>,
        views_confidences: LegacyMap::<felt252, u32>,
        N: u32, // num tickers
        K: u32, // num views
        risk_aversion: u32,
        tau: u32,
        risk_free: u32,
        cov_matrix: Array<Tensor<FixedType>>,
        decay_factor: u32,
        window: u32,
        bounds: (u32, u32),
        Q: Tensor::<FixedType>,
        P: Tensor::<FixedType>,
        pi: Tensor::<FixedType>,
        omega: Tensor::<FixedType>,
        posterior_returns: Tensor::<FixedType>,
        posterior_covariance: Tensor::<FixedType>,
        optimal_weights: LegacyMap::<felt252, Tensor::<FixedType>>,
        long_only: bool
    }

    // TODO: build constructor (need to set N, K, grab tickers/views from L1 contract, build covariance matrix...)

    // Black-Litterman Model Parameters

    #[view]
    fn parse_views() -> Tensor::<FixedType> {
        // TODO: Return Kx1 vector of views ("Q")
        let mut Q = Tensor::<FixedType>::new(); // it's possible inputs will be u32 and need to be converted to fixed type
        let mut Q_shape = Array::<u32>::new();
        Q_shape.append(self.K);
        let mut Q_data = Array::<FixedType>::new();
        let mut i: u32 = 0;
        loop {
            if i == self.K {
                break ();
            }
            Q_data.append(self.views.read(*self.tickers.at(i)));
            i += 1;
        };        
        let mut Q_extra = Option::<ExtraParams>::None(());
        let mut Q = TensorTrait::<FixedType>::new(Q_shape.span(), Q_data.span(), Q_extra);

        // Security check
        assert(*Q.shape.at(0)) = K;
        return Q;
    }

    #[view]
    fn parse_picks() -> Tensor::<FixedType> {
        // TODO: Return KxN matrix of ("P")
        // Since each ticker needs to map to a view, we will assume 0 represents no view
        // Sample views for ['BTC', 'ETH', 'UNI', 'BAT']:  
        // {'BTC': (0.02, 'ETH'), 'UNI': (0.03, '')}
        // -> [[ 1., -1.,  0.,  0.], [ 0.,  0.,  1.,  0.]]
        let mut P = Tensor::<FixedType>::new(); // it's possible inputs will be u32 and need to be converted to fixed type
        let mut P_shape = Array::<u32>::new();
        P_shape.append(self.K);
        P_shape.append(self.N);
        let mut P_data = Array::<FixedType>::new();

        let mut i: u32 = 0;
        loop {
            if i == self.K {
                break ();
            }
            // i = 0
            let mut view_ticker_i = self.views_tickers.read()[i]; // 'BTC'
            let mut view_ticker_i_index = self.get_ticker_index(view_ticker_i, self.tickers);
            let mut view_i = self.views.read(view_ticker_i)[0]; // 0.02
            let mut view_rel_ticker_i = self.views.read(view_ticker_i)[1]; // 'ETH'
            let mut j: u32 = 0;
            if view_rel_ticker_i == '' {
                loop {
                    if j == self.N {
                        break ();
                    }
                    else if j == view_ticker_i_index {
                        P_data.append(FixedTrait::new(8388608, false)); // 1 in Q8.23
                    }
                    else {
                        P_data.append(FixedTrait::new(0, false));
                    }
                    j += 1;
                };
            }
            else {
                let mut view_rel_ticker_i_index = self.get_ticker_index(view_rel_ticker_i, self.tickers);
                loop {
                    if j == self.N {
                        break ();
                    }
                    else if j == view_ticker_i_index {
                        P_data.append(FixedTrait::new(8388608, false)); // 1 in Q8.23
                    }
                    else if j == view_rel_ticker_i_index {
                        P_data.append(FixedTrait::new(8388608, true)); // -1 in Q8.23
                    }
                    else {
                        P_data.append(FixedTrait::new(0, false));
                    }
                    j += 1;
                };
            }
            i += 1;
        };

        let mut P_extra = Option::<ExtraParams>::None(());
        let mut P = TensorTrait::<FixedType>::new(P_shape.span(), P_data.span(), P_extra);

        // Security check
        assert(*P.shape.at(0)) = K;
        assert(*P.shape.at(1)) = N;
        return P;

    }

    #[view]
    fn prior_returns(market_caps: Tensor::<FixedType>) -> Tensor::<FixedType> {
        let mut market_caps = self.get_market_caps();
        let mut market_weights = market_caps / market_caps.sum();
        let mut pi = self.risk_aversion * self.cov_matrix.matmul(@market_weights);
        
        let pi_target_shape = ArrayTrait::new();
        pi_target_shape.append(1);
        let mut pi = pi.reshape(target_shape: pi_target_shape);

        assert(*pi.shape.at(0) = self.N);
        assert(*pi.shape.at(1) = 1);
        return pi;
    }

    #[view]
    fn build_omega() -> Tensor::<FixedType> {
        // TODO: build KxK diagonal "uncertainty" matrix
        let mut i = 0;
        let mut view_omegas = ArrayTrait::new();
        loop {
            if i == self.K {
                break ();
            }
            let mut view_i = *self.views.at(i);
            let mut view_confidence_i = self.views_confidences.read(view_i);
            let mut P_view_i = *self.P.at(i); // [1, -1, 0, 0]
            let mut alpha_i = (1 - view_confidence_i) / view_confidence_i;
            let mut omega_i_a = P_view_i.matmul(self.cov_matrix);
            let mut omega_i_b = omega_i_a.matmul(P_view_i.transpose()); // This should return a scalar. Does transpose need an argument?
            let mut omega_i = self.tau * alpha_i * omega_i_b;
            view_omegas.append(omega_i);
        };
        let mut omega = diagonalize(view_omegas);
        return omega;
    }

    // Posterior Estimates

    #[view]
    fn posterior_returns() -> Tensor::<FixedType> {
        // TODO: build Nx1 posterior estimate of returns
        let mut tau_sigma_P = self.tau * self.cov_matrix.matmul(self.P.transpose());
        let mut A = self.P.matmul(tau_sigma_P) + self.omega;
        let mut b = self.Q - self.P.matmul(self.pi);
        let mut solution = linalg_solve(A, b);
        let mut post_rets = self.pi + tau_sigma_P.matmul(solution);
        return post_rets;
    }

    #[view]
    fn posterior_covariance() -> Tensor::<FixedType> {
        // TODO: build NxN posterior estimate of covariance matrix
        let mut tau_sigma_P = self.tau * self.cov_matrix.matmul(self.P.transpose());
        let mut A = self.P.matmul(tau_sigma_P) + self.omega;
        let mut b = tau_sigma_P.transpose();
        let mut M = self.tau * self.cov_matrix - tau_sigma_P.matmul(linalg_solve(A, b));
        let mut posterior_covariance = self.cov_matrix + M;
        return posterior_covariance;
    }

    // Optimal Weights

    #[external]
    fn optimal_weights(long_only: bool) -> Felt252Dict<u32> {
        // TODO: calculate dictionary of optimal weights
        let mut A = self.risk_aversion * self.cov_matrix;
        let mut raw_weights = linalg_solve(A, self.posterior_returns);
        let mut weights = raw_weights / raw_weights.sum();
        let mut weights_dict: Felt252Dict<u64> = Default::default();

        let mut t = 0;
        loop {
            if t == self.N {
                break ();
            }
            weights_dict.insert(*self.tickers.at(i), *weights.at(i));
            t += 1;
        };

        // TODO: convert optimal weights to positive values by normalizing against absolute gross sum
        return weights_dict;
    }

    // Helper Functions

    #[view]
    fn get_ticker_index(ticker: felt252, tickers: Array::<felt252>) -> u32 {
        let mut i: u32 = 0;
        loop {
            if ticker == *tickers.at(i) {
                break ();                
            }
            i += 1;
        };
        return i;
    }

    #[view]
    fn get_market_caps() -> Tensor::<FixedType> {
        // TODO: get market caps
    }

}