My graph neural network was the most accurate traffic predictor I built. It was also blind to a sensor that simply froze.

The question: when a loop detector suddenly reports a jam, is it real, or is the sensor tampered with? Real congestion shows up at neighbouring sensors too. A lying sensor stands alone. So I trained models to predict each sensor's speed from the network around it, and flagged big disagreements.

Data: real Caltrans PEMS-BAY, 325 sensors over 6 months, with 150 tampering events injected into May-June.

What worked:
- Fake jams and noisy sensors: 93-100% caught, mostly within minutes, at 0.16% false alarms

What didn't:
- Frozen values, slow drift, and yesterday's data replayed: 0-7% caught by every model that used the sensor's own history

Why: once a sensor starts lying, its own recent history is a lie too. A frozen sensor predicts itself perfectly. A better forecaster doesn't fix that; the design does.

So I built a version that predicts each sensor only from its neighbours. Fake jams went to 100% and drift to 37%. But it's less precise, so it missed noisy sensors and tripled false alarms in real congestion. No single detector wins.

Repo: https://github.com/faysalchowdhoury/arterial-sensor-tamper-gnn

How do TMCs validate detector data before using it for ramp metering or incident detection?

#TrafficData #GraphNeuralNetworks #AnomalyDetection #Cybersecurity #ITS #TransportationEngineering #Python
