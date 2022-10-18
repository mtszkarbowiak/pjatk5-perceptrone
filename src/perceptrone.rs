extern crate rand;
use rand::*;

pub struct Perceptrone<const DIM : usize> {
    weights : [f32; DIM],
    threshold : f32,
}

impl<const DIM : usize> Perceptrone<DIM> {
    pub fn new<const DIM2 : usize>(random_weights_scale : Option<f32>) -> Perceptrone<DIM> {
        let mut weights = [0.0; DIM];
        let mut threshold = 0.0;

        if random_weights_scale.is_some() {
            let mut rng = thread_rng();
            for i in 0..DIM {
                weights[i] = rng.gen_range(0.0, random_weights_scale.unwrap());
            }
            threshold = rng.gen_range(0.0, random_weights_scale.unwrap());
        }

        return Perceptrone{ weights, threshold };
    }

    pub fn mul(&self, input : [f32; DIM]) -> f32 {
        let mut result = 0.0;
        for d in 0..DIM {
            result += self.weights[d] * input[d];
        }
        return result;
    }

    pub fn classify(&self, input : [f32; DIM]) -> bool {
        self.mul(input) >= self.threshold
    }

    pub fn learn(&mut self, 
        input : [f32; DIM], 
        learn_speed : f32, 
        valid_result : bool
    ) -> bool {
        assert!(0.0 < learn_speed && learn_speed < 1.0);

        let calculated_result = self.classify(input);
        let result_is_valid = valid_result == calculated_result;
        let result_sign = if result_is_valid { 1.0 } else { -1.0 };

        for d in 0..DIM {
            self.weights[d] += result_sign * learn_speed * input[d];
        }
        self.threshold += result_sign * learn_speed * (-1.0);
        return result_is_valid;
    }

    pub fn print(&self) {
        print!("Threshold: {:.2} Weights: ", self.threshold);
        for d in 0..DIM {
            print!(" {:.2}", self.weights[d]);
        }
        println!("");
    }
}