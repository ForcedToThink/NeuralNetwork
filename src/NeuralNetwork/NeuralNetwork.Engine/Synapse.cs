using System;

namespace NeuralNetwork.Engine
{
    public class Synapse : ISynapse
    {
        internal INeuron _fromNeuron;
        internal INeuron _toNeuron;

        public double Weight { get; set; }
        public double PreviousWeight { get; set; }

        public Synapse(INeuron fromNeuron, INeuron toNeuron)
        {
            this._fromNeuron = fromNeuron;
            this._toNeuron = toNeuron;

            var tmpRandom = new Random();
            this.Weight = tmpRandom.NextDouble();
            this.PreviousWeight = 0;
        }

        public Synapse(INeuron fromNeuron, INeuron toNeuron, double weight)
        {
            this._fromNeuron = fromNeuron;
            this._toNeuron = toNeuron;

            this.Weight = weight;
            this.PreviousWeight = 0;
        }

        public double GetOutput()
        {
            return this._fromNeuron.CalculateOutput();
        }

        public bool IsFromNeuron(Guid fromNeuronId)
        {
            return this._fromNeuron.Id.Equals(fromNeuronId);
        }

        public void UpdateWeight(double learningRate, double delta)
        {
            this.PreviousWeight = this.Weight;
            this.Weight += learningRate * delta;
        }
    }
}