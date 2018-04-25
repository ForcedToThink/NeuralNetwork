using System;

namespace NeuralNetwork.Engine.ActvationFunctions
{
    public class StepActivationFunction : IActivationFunction
    {
        private readonly double _treshold;

        public StepActivationFunction(double treshold)
        {
            this._treshold = treshold;
        }

        public double CalculateOutput(double input)
        {
            return Convert.ToDouble(input > this._treshold);
        }
    }
}