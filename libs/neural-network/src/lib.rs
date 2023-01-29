use nalgebra::base::{DMatrix, Matrix, VecStorage};
use nalgebra::Dyn;
use rand::Rng;

use std::iter::once;

#[derive(Debug)]
struct Network {
    layers: Vec<Layer>,
}

#[derive(Debug, Clone)]
struct Layer {
    neurons: Vec<Neuron>,
}

#[derive(Clone)]
struct LayerTopology {
    neurons: usize,
}

#[derive(Debug, Clone)]
struct Neuron {
    weights: Vec<f32>,
    bias: f32,
}

impl Network {
    fn new(layers: Vec<Layer>) -> Network {
        Self { layers }
    }

    fn weights(&self) -> Vec<f32> {
        let mut layers = self.layers.clone();
        layers.remove(0);

        layers
            .iter()
            .flat_map(|layer| layer.neurons.iter())
            .flat_map(|neuron| once(&neuron.bias).chain(&neuron.weights))
            .cloned()
            .collect()
    }

    fn from_weights(weights: Vec<f32>, layers_topology: Vec<LayerTopology>) -> Network {
        let mut weights = weights.into_iter();

        let mut working_topology: Vec<LayerTopology> = Vec::new();
        working_topology.push(LayerTopology::new(0));
        for layer in layers_topology.iter() {
            working_topology.push(layer.to_owned());
        }
        working_topology.push(LayerTopology::new(0));

        let mut layers: Vec<Layer> = working_topology
            .windows(2)
            .enumerate()
            .map(|(index, layer)| {
                Layer::from_weights(&mut weights, layer[1].neurons, layer[0].neurons, index == 0)
            })
            .collect();

        layers.remove(layers.len() - 1);

        Network { layers }
    }

    fn random(rng: &mut dyn rand::RngCore, layers_topology: Vec<LayerTopology>) -> Network {
        assert!(layers_topology.len() > 1);

        let mut working_topology: Vec<LayerTopology> = Vec::new();
        working_topology.push(LayerTopology::new(0));
        for layer in layers_topology.iter() {
            working_topology.push(layer.to_owned());
        }
        working_topology.push(LayerTopology::new(0));

        let mut layers: Vec<Layer> = working_topology
            .windows(2)
            .enumerate()
            .map(|(index, layer)| {
                Layer::random(rng, layer[1].neurons, layer[0].neurons, index == 0)
            })
            .collect();

        layers.remove(layers.len() - 1);

        Self { layers }
    }

    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        assert_eq!(self.layers[0].neurons.len(), inputs.len());

        let mut layers_weights: Vec<Vec<Vec<f32>>> =
            self.layers.iter().map(|layer| layer.weights()).collect();
        let mut layers_bias: Vec<Vec<f32>> = self.layers.iter().map(|layer| layer.bias()).collect();

        layers_weights.remove(0);
        layers_bias.remove(0);

        let output = layers_weights.iter().zip(layers_bias.iter()).fold(
            inputs,
            |inputs, (weights, bias)| {
                let weights_matrix = DMatrix::from_vec(
                    weights.len(),
                    weights[0].len(),
                    weights
                        .iter()
                        .map(|weight| weight.to_owned())
                        .flatten()
                        .collect::<Vec<f32>>(),
                );
                let bias_matrix = DMatrix::from_vec(bias.len(), 1, bias.to_owned());
                let inputs_matrix = DMatrix::from_vec(inputs.len(), 1, inputs);

                let z_matrix = (weights_matrix * &inputs_matrix) + bias_matrix;
                let sigmoid_z = z_matrix
                    .iter()
                    .map(|e| sigmoid(e.clone()))
                    .collect::<Vec<f32>>();

                sigmoid_z
            },
        );

        output
    }

    fn costs(&self, inputs: Vec<f32>, expected: Vec<f32>) -> f32 {
        let results = self.propagate(inputs);
        assert_eq!(results.len(), expected.len());

        results
            .iter()
            .zip(expected.iter())
            .map(|(result, expectation)| (result - expectation).powf(2.0))
            .sum()
    }
}

impl Layer {
    fn new(neurons: Vec<Neuron>) -> Layer {
        Self { neurons }
    }

    fn random(
        rng: &mut dyn rand::RngCore,
        neurons_size: usize,
        weights_size: usize,
        input_layer: bool,
    ) -> Layer {
        let neurons = (0..neurons_size)
            .map(|_| Neuron::random(rng, weights_size, input_layer))
            .collect();

        Self { neurons }
    }

    fn weights(&self) -> Vec<Vec<f32>> {
        self.neurons
            .iter()
            .map(|neuron| &neuron.weights)
            .cloned()
            .collect::<Vec<Vec<f32>>>()
    }

    fn bias(&self) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| neuron.bias)
            .collect::<Vec<f32>>()
    }

    fn weights_matrix(&self) -> Matrix<f32, Dyn, Dyn, VecStorage<f32, Dyn, Dyn>> {
        DMatrix::from_vec(
            self.neurons.len(),
            self.neurons[0].weights.len(),
            self.weights()
                .iter()
                .map(|weight| weight.to_owned())
                .flatten()
                .collect::<Vec<f32>>(),
        )
    }

    fn bias_matrix(&self) -> Matrix<f32, Dyn, Dyn, VecStorage<f32, Dyn, Dyn>> {
        DMatrix::from_vec(self.neurons.len(), 1, self.bias())
    }

    fn from_weights(
        weights: &mut std::vec::IntoIter<f32>,
        neurons_size: usize,
        weights_size: usize,
        input_layer: bool,
    ) -> Layer {
        let neurons = (0..neurons_size)
            .map(|_| Neuron::from_weights(weights, weights_size, input_layer))
            .collect();

        Self { neurons }
    }
}

impl LayerTopology {
    fn new(neurons: usize) -> LayerTopology {
        Self { neurons }
    }
}

impl Neuron {
    fn new(weights: Vec<f32>, bias: f32) -> Neuron {
        Self { weights, bias }
    }

    fn from_weights(
        weights: &mut std::vec::IntoIter<f32>,
        weights_size: usize,
        input_neuron: bool,
    ) -> Neuron {
        let (weights, bias) = if input_neuron {
            (vec![0.0], 0.0)
        } else {
            let bias = weights.next().expect("not enough weights");
            let weights = (0..weights_size)
                .map(|_| weights.next().expect("not enough weights"))
                .collect();

            (weights, bias)
        };

        Self { weights, bias }
    }

    fn random(rng: &mut dyn rand::RngCore, weights_size: usize, input_neuron: bool) -> Neuron {
        let (weights, bias) = if input_neuron {
            (vec![0.0], 0.0)
        } else {
            let random_bias = rng.gen_range(-1.0..=1.0);
            let random_weights = (0..weights_size)
                .map(|_| rng.gen_range(-1.0..=1.0))
                .collect();

            (random_weights, random_bias)
        };

        Self { weights, bias }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-x))
}

fn invsigmoid(x: f32) -> f32 {
   (x / (1.0 - x)).ln() 
}

#[cfg(test)]
mod tests {
    mod test_sigmoid {
        use crate::{sigmoid, invsigmoid};

        #[test]
        fn test() {
            assert_eq!(sigmoid(0.0), 0.5);
            assert_eq!(sigmoid(-10.0), 4.5397872e-5);
            assert_eq!(sigmoid(10.0), 0.9999546);

	    assert_eq!(sigmoid(invsigmoid(0.5)), 0.5);
        }
    }

    mod test_matrix {
        use nalgebra::base::DMatrix;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        use crate::Layer;

        #[test]
        fn test() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let layer = Layer::random(&mut rng, 3, 3, false);

            let weights_matrix = DMatrix::from_vec(
                3,
                3,
                vec![
                    0.67383957,
                    0.8181262,
                    0.26284897,
                    -0.53516835,
                    0.069369674,
                    -0.7648182,
                    -0.48879617,
                    -0.19277132,
                    -0.8020501,
                ],
            );

            let bias_matrix = DMatrix::from_vec(3, 1, vec![-0.6255188, 0.5238807, -0.102499366]);

            assert_eq!(layer.weights_matrix(), weights_matrix);
            assert_eq!(layer.bias_matrix(), bias_matrix);
        }
    }

    mod test_from_weights {
        use crate::{LayerTopology, Network};
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        #[test]
        fn test() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let network1 = Network::random(
                &mut rng,
                vec![
                    LayerTopology::new(2),
                    LayerTopology::new(3),
                    LayerTopology::new(2),
                ],
            );

            let weights = vec![
                -0.6255188,
                0.67383957,
                0.8181262,
                0.26284897,
                0.5238807,
                -0.53516835,
                0.069369674,
                -0.7648182,
                -0.102499366,
                -0.48879617,
                -0.19277132,
                -0.8020501,
                0.2754606,
                -0.98680043,
                0.4452356,
                -0.47662205,
                -0.89078736,
            ];

            let network2 = Network::from_weights(
                weights,
                vec![
                    LayerTopology::new(2),
                    LayerTopology::new(3),
                    LayerTopology::new(2),
                ],
            );

            assert_eq!(network1.weights(), network2.weights());
            assert_eq!(
                network1.propagate(vec![0.7538273, 0.1487342]),
                network2.propagate(vec![0.7538273, 0.1487342])
            );
        }
    }

    mod test_propagate {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        use crate::{LayerTopology, Network};

        #[test]
        fn test() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let network = Network::random(
                &mut rng,
                vec![
                    LayerTopology::new(2),
                    LayerTopology::new(5),
                    LayerTopology::new(5),
                    LayerTopology::new(1),
                ],
            );

            let inputs = vec![0.2983741, 0.4878372];
            let output = vec![0.6991053];

            assert_eq!(network.propagate(inputs), output);
        }
    }

    mod test_costs {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        use crate::{LayerTopology, Network};

        #[test]
        fn test() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let network = Network::random(
                &mut rng,
                vec![
                    LayerTopology::new(2),
                    LayerTopology::new(5),
                    LayerTopology::new(5),
                    LayerTopology::new(3),
                ],
            );

            let inputs = vec![0.2983741, 0.4878372];
            let expected = vec![0.6991053, 0.8732371, 0.3938746];

            assert_eq!(network.costs(inputs, expected), 0.24041694);
        }
    }
}
