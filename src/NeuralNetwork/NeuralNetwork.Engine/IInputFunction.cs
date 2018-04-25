using System.Collections.Generic;

namespace NeuralNetwork.Engine
{
    public interface IInputFunction
    {
        double CalculateInput(List<ISynapse> inputs);
    }
}