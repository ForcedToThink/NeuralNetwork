using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Engine;
using NeuralNetwork.Engine.ActvationFunctions;
using NeuralNetwork.Engine.InputFunctions;

namespace UI
{
    class Program
    {
        static void Main(string[] args)
        {
            var network = new SimpleNeuralNetwork(3);

            network.AddLayer(NeuralLayerFactory.CreateNeuralLayer(3, new RectifiedActivationFunction(), new WeightedSumFunction()));
            network.AddLayer(NeuralLayerFactory.CreateNeuralLayer(1, new SigmoidActivationFunction(0.7), new WeightedSumFunction()));

            network.PushExpectedValues(
                new double[][]
                {
                    new double[] {0},
                    new double[] {1},
                    new double[] {1},
                    new double[] {0},
                    new double[] {1},
                    new double[] {0},
                    new double[] {0},
                });

            network.Train(new double[][]
            {
                new double[] { 150, 2, 0 },
                new double[] { 1002, 56, 1 },
                new double[] { 1060, 59, 1 }, 
                new double[] { 200, 3, 0 },
                new double[] { 300, 3, 1 },
                new double[] { 120, 1, 0 },
                new double[] { 80, 1, 0 }
            }, 10000);

            network.PushInputValues(new double[] { 1060, 59, 1 });
            var outputs = network.GetOutput();

            Console.WriteLine($"{outputs[0]}");

            Console.ReadKey();
        }
    }
}
