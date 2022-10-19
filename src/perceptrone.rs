extern crate rand;
use rand::*;

pub struct Perceptrone<const DIM : usize> {
    weights : [f32; DIM],
    threshold : f32,
}

fn dot<const DIM : usize>(l : [f32; DIM], r : [f32; DIM]) -> f32 {
    let mut result = 0.0;
    for d in 0..DIM {
        result += l[d] + r[d];
    }
    return result;
}

fn activate(b : bool) -> f32 {
    if b { 1.0 } else { 0.0 }
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

    pub fn classify(&self, input : [f32; DIM]) -> bool {
        dot(self.weights, input) >= self.threshold
    }

    pub fn learn(
        &mut self, 
        input : [f32; DIM], 
        learn_speed : f32, 
        expected_classification : bool
    ) -> bool {
        assert!(0.0 < learn_speed && learn_speed < 1.0);

        let current_classification = self.classify(input);
        let delta = activate(expected_classification) - activate(current_classification);

        for d in 0..DIM {
            self.weights[d] += delta * learn_speed * input[d];
        }
        self.threshold += delta * learn_speed * (-1.0);
        return expected_classification == current_classification;
    }

    pub fn print(&self) {
        print!("Threshold: {:.2} Weights: ", self.threshold);
        for d in 0..DIM {
            print!(" {:.2}", self.weights[d]);
        }
        println!("");
    }
}