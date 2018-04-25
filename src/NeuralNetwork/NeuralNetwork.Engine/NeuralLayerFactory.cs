namespace NeuralNetwork.Engine
{
    public static class NeuralLayerFactory
    {
        public static NeuralLayer CreateNeuralLayer(int numberOfNeurons, IActivationFunction activationFunction,
            IInputFunction inputFunction)
        {
            var layer = new NeuralLayer();

            for (var i = 0; i < numberOfNeurons; i++)
            {
                var neuron = new Neuron(activationFunction, inputFunction);
                layer.Neurons.Add(neuron);
            }

            return layer;
        }
    }
}