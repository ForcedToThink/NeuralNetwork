using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Engine.ActvationFunctions;
using NeuralNetwork.Engine.InputFunctions;

namespace NeuralNetwork.Engine
{
    public class SimpleNeuralNetwork
    {
        internal List<NeuralLayer> _layers;
        internal double _learningRate;
        internal double[][] _expectedResults;

        public SimpleNeuralNetwork(int numberOfInputNeurons)
        {
            _layers = new List<NeuralLayer>();

            CreateInputLayer(numberOfInputNeurons);

            _learningRate = 2.95;
        }

        public void AddLayer(NeuralLayer newLayer)
        {
            if (_layers.Any())
            {
                var lastLayer = _layers.Last();
                newLayer.ConnectLayers(lastLayer);
            }

            _layers.Add(newLayer);
        }

        public void PushInputValues(double[] inputs)
        {
            _layers.First().Neurons.ForEach(x => x.PushValueOnInput(inputs[_layers.First().Neurons.IndexOf(x)]));
        }

        public void PushExpectedValues(double[][] expectedValues)
        {
            _expectedResults = expectedValues;
        }

        public List<double> GetOutput()
        {
            var returnValue = new List<double>();
            
            _layers.Last().Neurons.ForEach(neuron =>
            {
                returnValue.Add(neuron.CalculateOutput());
            });

            return returnValue;
        }

        public void Train(double[][] inputs, int numberOfEpochs)
        {
            double totalError = 0;

            for (var i = 0; i < numberOfEpochs; i++)
            {
                for (var j = 0; j < inputs.GetLength(0); j++)
                {
                    PushInputValues(inputs[j]);

                    var outputs = new List<double>();

                    _layers.Last().Neurons.ForEach(x =>
                    {
                        outputs.Add(x.CalculateOutput());
                    });

                    totalError = CalculateTotalError(outputs, j);
                    HandleOutputLayer(j);
                    HandleHiddenLayers();
                }
            }
        }

        private void CreateInputLayer(int numberOfInputs)
        {
            var inputLayer = NeuralLayerFactory.CreateNeuralLayer(numberOfInputs, new RectifiedActivationFunction(),
                new WeightedSumFunction());
            inputLayer.Neurons.ForEach(x => x.AddInputSynapse(0));
            this.AddLayer(inputLayer);
        }

        private double CalculateTotalError(List<double> outputs, int row)
        {
            double totalError = 0;
            outputs.ForEach(output =>
            {
                var error = Math.Pow(output - _expectedResults[row][outputs.IndexOf(output)], 2);
                totalError += error;
            });

            return totalError;
        }

        private void HandleOutputLayer(int row)
        {
            _layers.Last().Neurons.ForEach(neuron =>
            {
                neuron.Inputs.ForEach(connection =>
                {
                    var output = neuron.CalculateOutput();
                    var netInput = connection.GetOutput();

                    var expectedOutput = _expectedResults[row][_layers.Last().Neurons.IndexOf(neuron)];

                    var nodeDelta = (expectedOutput - output) * output * (1 - output);
                    var delta = -1 * netInput * nodeDelta;

                    connection.UpdateWeight(_learningRate, delta);

                    neuron.PreviousPartialDerivate = nodeDelta;
                });
            });
        }

        private void HandleHiddenLayers()
        {
            for (var k = _layers.Count - 2; k > 0; k--)
            {
                _layers[k].Neurons.ForEach(neuron =>
                {
                    neuron.Inputs.ForEach(connection =>
                    {
                        var output = neuron.CalculateOutput();
                        var netInput = connection.GetOutput();
                        double sumPartial = 0;

                        _layers[k + 1].Neurons.ForEach(outputNeuron =>
                        {
                            outputNeuron.Inputs.Where(i => i.IsFromNeuron(neuron.Id))
                                .ToList()
                                .ForEach(outConnection =>
                                {
                                    sumPartial += outConnection.PreviousWeight *
                                                  outputNeuron.PreviousPartialDerivate;
                                });
                        });

                        var delta = -1 * netInput * sumPartial * output * (1 - output);
                        connection.UpdateWeight(_learningRate, delta);
                    });
                });
            }
        }
    }
}