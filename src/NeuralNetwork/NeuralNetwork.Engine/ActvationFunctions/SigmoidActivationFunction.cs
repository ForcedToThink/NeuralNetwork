using System;
using System.Text.RegularExpressions;

namespace NeuralNetwork.Engine.ActvationFunctions
{
    public class SigmoidActivationFunction : IActivationFunction
    {
        private readonly double _coeficient;

        public SigmoidActivationFunction(double coeficient)
        {
            this._coeficient = coeficient;
        }

        public double CalculateOutput(double input)
        {
            return 1 / (1 + Math.Exp(-input * _coeficient));
        }
    }
}