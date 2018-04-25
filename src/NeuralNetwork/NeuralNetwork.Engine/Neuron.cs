using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.Engine
{
    public class Neuron : INeuron
    {
        private IActivationFunction _activationFunction;
        private IInputFunction _inputFunction;

        public Guid Id { get; private set; }
        public double PreviousPartialDerivate { get; set; }
        public List<ISynapse> Inputs { get; set; }
        public List<ISynapse> Outputs { get; set; }

        public Neuron(IActivationFunction activationFunction, IInputFunction inputFunction)
        {
            this.Id = Guid.NewGuid();
            this.Inputs = new List<ISynapse>();
            this.Outputs = new List<ISynapse>();

            this._activationFunction = activationFunction;
            this._inputFunction = inputFunction;
        }

        public void AddInputNeuron(INeuron inputNeuron)
        {
            var synapse = new Synapse(inputNeuron, this);
            this.Inputs.Add(synapse);
            inputNeuron.Outputs.Add(synapse);
        }

        public void AddOutputNeuron(INeuron outputNeuron)
        {
            var synapse = new Synapse(this, outputNeuron);
            this.Outputs.Add(synapse);
            outputNeuron.Inputs.Add(synapse);
        }

        public double CalculateOutput()
        {
            return this._activationFunction.CalculateOutput(this._inputFunction.CalculateInput(this.Inputs));
        }

        public void AddInputSynapse(double inputValue)
        {
            var inputSynapse = new InputSynapse(this, inputValue);
            this.Inputs.Add(inputSynapse);
        }

        public void PushValueOnInput(double inputValue)
        {
            ((InputSynapse) Inputs.First()).Output = inputValue;
        }
    }
}