package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"syscall/js"
	"time"
)

type Activation string

const (
	ActivationBinary   Activation = "binary"
	ActivationLogistic Activation = "logistic"
	ActivationReLU     Activation = "relu"
)

type NetworkConfig struct {
	Layers       []int    `json:"layers"`
	Activations  []string `json:"activations"`
	LearningRate float64  `json:"learning_rate,omitempty"`
	Seed         *int64   `json:"seed,omitempty"`
	WeightScale  float64  `json:"weight_scale,omitempty"`
}

type Sample struct {
	Input  []float64 `json:"input"`
	Target []float64 `json:"target"`
}

type TrainRequest struct {
	State        NetworkState `json:"state"`
	Dataset      []Sample     `json:"dataset"`
	Epochs       int          `json:"epochs"`
	LearningRate *float64     `json:"learning_rate,omitempty"`
	Shuffle      bool         `json:"shuffle,omitempty"`
}

type ForwardRequest struct {
	State NetworkState `json:"state"`
	Input []float64    `json:"input"`
}

type TrainResponse struct {
	State       NetworkState `json:"state"`
	LossHistory []float64    `json:"loss_history"`
	FinalLoss   float64      `json:"final_loss"`
	Samples     int          `json:"samples"`
	Epochs      int          `json:"epochs"`
}

type ForwardResponse struct {
	Output []float64 `json:"output"`
}

type TrainerInitRequest struct {
	State        NetworkState `json:"state"`
	Dataset      []Sample     `json:"dataset"`
	LearningRate *float64     `json:"learning_rate,omitempty"`
	Shuffle      bool         `json:"shuffle,omitempty"`
}

type TrainerCommandRequest struct {
	TrainerID string `json:"trainer_id"`
}

type TrainerStatusResponse struct {
	TrainerID    string       `json:"trainer_id"`
	Running      bool         `json:"running"`
	EpochsDone   int          `json:"epochs_done"`
	LossHistory  []float64    `json:"loss_history"`
	FinalLoss    float64      `json:"final_loss"`
	HasFinalLoss bool         `json:"has_final_loss"`
	Deviation    float64      `json:"deviation"`
	State        NetworkState `json:"state"`
	Error        string       `json:"error,omitempty"`
}

type Network struct {
	weights      [][][]float64
	biases       [][]float64
	activations  []Activation
	learningRate float64
}

type NetworkState struct {
	Layers       []int         `json:"layers"`
	Activations  []string      `json:"activations"`
	LearningRate float64       `json:"learning_rate"`
	Weights      [][][]float64 `json:"weights"`
	Biases       [][]float64   `json:"biases"`
}

var (
	callbacks []js.Func
	trainers  = map[string]*Trainer{}
	trainersM sync.Mutex
)

type Trainer struct {
	id           string
	net          *Network
	dataset      []Sample
	learningRate *float64
	shuffle      bool

	mu          sync.Mutex
	running     bool
	stop        bool
	epochsDone  int
	lossHistory []float64
	finalLoss   float64
	hasFinal    bool
	deviation   float64
	lastErr     string
}

func clone3D(src [][][]float64) [][][]float64 {
	dst := make([][][]float64, len(src))
	for i := range src {
		dst[i] = make([][]float64, len(src[i]))
		for j := range src[i] {
			dst[i][j] = append([]float64(nil), src[i][j]...)
		}
	}
	return dst
}

func clone2D(src [][]float64) [][]float64 {
	dst := make([][]float64, len(src))
	for i := range src {
		dst[i] = append([]float64(nil), src[i]...)
	}
	return dst
}

func (n *Network) ToState() NetworkState {
	layers := make([]int, len(n.weights)+1)
	if len(n.weights) > 0 && len(n.weights[0]) > 0 {
		layers[0] = len(n.weights[0][0])
	}
	for layer := 0; layer < len(n.weights); layer++ {
		layers[layer+1] = len(n.weights[layer])
	}

	acts := make([]string, len(n.activations))
	for i, a := range n.activations {
		acts[i] = string(a)
	}

	return NetworkState{
		Layers:       layers,
		Activations:  acts,
		LearningRate: n.learningRate,
		Weights:      clone3D(n.weights),
		Biases:       clone2D(n.biases),
	}
}

func networkFromState(state NetworkState) (*Network, error) {
	if len(state.Layers) < 2 {
		return nil, fmt.Errorf("state.layers muss mindestens zwei Schichten enthalten")
	}
	for i, size := range state.Layers {
		if size <= 0 {
			return nil, fmt.Errorf("state.layers[%d] muss > 0 sein", i)
		}
	}

	if len(state.Activations) != len(state.Layers)-1 {
		return nil, fmt.Errorf("state.activations muss genau %d Einträge haben", len(state.Layers)-1)
	}

	acts := make([]Activation, len(state.Activations))
	for i, raw := range state.Activations {
		act, err := normalizeActivation(raw)
		if err != nil {
			return nil, err
		}
		acts[i] = act
	}

	if len(state.Weights) != len(state.Layers)-1 {
		return nil, fmt.Errorf("state.weights hat %d Layer statt %d", len(state.Weights), len(state.Layers)-1)
	}
	if len(state.Biases) != len(state.Layers)-1 {
		return nil, fmt.Errorf("state.biases hat %d Layer statt %d", len(state.Biases), len(state.Layers)-1)
	}

	for layer := 0; layer < len(state.Layers)-1; layer++ {
		inSize := state.Layers[layer]
		outSize := state.Layers[layer+1]

		if len(state.Weights[layer]) != outSize {
			return nil, fmt.Errorf("state.weights[%d] hat %d Neuronen statt %d", layer, len(state.Weights[layer]), outSize)
		}
		if len(state.Biases[layer]) != outSize {
			return nil, fmt.Errorf("state.biases[%d] hat %d Einträge statt %d", layer, len(state.Biases[layer]), outSize)
		}
		for node := 0; node < outSize; node++ {
			if len(state.Weights[layer][node]) != inSize {
				return nil, fmt.Errorf(
					"state.weights[%d][%d] hat %d Einträge statt %d",
					layer,
					node,
					len(state.Weights[layer][node]),
					inSize,
				)
			}
		}
	}

	lr := state.LearningRate
	if lr == 0 {
		lr = 0.1
	}
	if lr <= 0 {
		return nil, fmt.Errorf("state.learning_rate muss > 0 sein")
	}

	return &Network{
		weights:      clone3D(state.Weights),
		biases:       clone2D(state.Biases),
		activations:  acts,
		learningRate: lr,
	}, nil
}

func normalizeActivation(raw string) (Activation, error) {
	switch raw {
	case "binary":
		return ActivationBinary, nil
	case "logistic", "sigmoid":
		return ActivationLogistic, nil
	case "relu":
		return ActivationReLU, nil
	default:
		return "", fmt.Errorf("unbekannte Aktivierungsfunktion: %s", raw)
	}
}

func activate(act Activation, x float64) float64 {
	switch act {
	case ActivationBinary:
		if x >= 0 {
			return 1
		}
		return 0
	case ActivationLogistic:
		return 1.0 / (1.0 + math.Exp(-x))
	case ActivationReLU:
		if x > 0 {
			return x
		}
		return 0
	default:
		return x
	}
}

func derivative(act Activation, x float64, activated float64) float64 {
	switch act {
	case ActivationBinary:
		// Straight-through Schätzer für binäre Aktivierung.
		if math.Abs(x) <= 1 {
			return 1
		}
		return 0
	case ActivationLogistic:
		return activated * (1.0 - activated)
	case ActivationReLU:
		if x > 0 {
			return 1
		}
		return 0
	default:
		return 1
	}
}

func newNetwork(cfg NetworkConfig) (*Network, error) {
	if len(cfg.Layers) < 2 {
		return nil, fmt.Errorf("layers muss mindestens Eingabe- und Ausgabeschicht enthalten")
	}

	for i, size := range cfg.Layers {
		if size <= 0 {
			return nil, fmt.Errorf("layers[%d] muss > 0 sein", i)
		}
	}

	if len(cfg.Activations) != len(cfg.Layers)-1 {
		return nil, fmt.Errorf("activations muss genau %d Einträge haben", len(cfg.Layers)-1)
	}

	acts := make([]Activation, len(cfg.Activations))
	for i, raw := range cfg.Activations {
		act, err := normalizeActivation(raw)
		if err != nil {
			return nil, err
		}
		acts[i] = act
	}

	lr := cfg.LearningRate
	if lr == 0 {
		lr = 0.1
	}
	if lr <= 0 {
		return nil, fmt.Errorf("learning_rate muss > 0 sein")
	}

	scale := cfg.WeightScale
	if scale == 0 {
		scale = 1.0
	}
	if scale <= 0 {
		return nil, fmt.Errorf("weight_scale muss > 0 sein")
	}

	seed := time.Now().UnixNano()
	if cfg.Seed != nil {
		seed = *cfg.Seed
	}
	rng := rand.New(rand.NewSource(seed))

	weights := make([][][]float64, len(cfg.Layers)-1)
	biases := make([][]float64, len(cfg.Layers)-1)

	for layer := 0; layer < len(cfg.Layers)-1; layer++ {
		inSize := cfg.Layers[layer]
		outSize := cfg.Layers[layer+1]

		weights[layer] = make([][]float64, outSize)
		biases[layer] = make([]float64, outSize)

		heScale := math.Sqrt(2.0/float64(inSize)) * scale
		for j := 0; j < outSize; j++ {
			weights[layer][j] = make([]float64, inSize)
			for i := 0; i < inSize; i++ {
				weights[layer][j][i] = (rng.Float64()*2.0 - 1.0) * heScale
			}
			biases[layer][j] = 0
		}
	}

	return &Network{
		weights:      weights,
		biases:       biases,
		activations:  acts,
		learningRate: lr,
	}, nil
}

func (n *Network) forwardWithCache(input []float64) ([][]float64, [][]float64, error) {
	if len(input) != len(n.weights[0][0]) {
		return nil, nil, fmt.Errorf("input-Größe %d passt nicht zur Eingabeschicht %d", len(input), len(n.weights[0][0]))
	}

	activations := make([][]float64, len(n.weights)+1)
	zs := make([][]float64, len(n.weights))

	activations[0] = append([]float64(nil), input...)

	for layer := 0; layer < len(n.weights); layer++ {
		outSize := len(n.weights[layer])
		z := make([]float64, outSize)
		a := make([]float64, outSize)

		for j := 0; j < outSize; j++ {
			sum := n.biases[layer][j]
			for i := 0; i < len(n.weights[layer][j]); i++ {
				sum += n.weights[layer][j][i] * activations[layer][i]
			}
			z[j] = sum
			a[j] = activate(n.activations[layer], sum)
		}

		zs[layer] = z
		activations[layer+1] = a
	}

	return activations, zs, nil
}

func (n *Network) Forward(input []float64) ([]float64, error) {
	activations, _, err := n.forwardWithCache(input)
	if err != nil {
		return nil, err
	}
	output := activations[len(activations)-1]
	return append([]float64(nil), output...), nil
}

func (n *Network) Train(dataset []Sample, epochs int, lrOverride *float64, shuffle bool) (TrainResponse, error) {
	if len(dataset) == 0 {
		return TrainResponse{}, fmt.Errorf("dataset darf nicht leer sein")
	}
	if epochs <= 0 {
		return TrainResponse{}, fmt.Errorf("epochs muss > 0 sein")
	}

	lr := n.learningRate
	if lrOverride != nil {
		if *lrOverride <= 0 {
			return TrainResponse{}, fmt.Errorf("learning_rate muss > 0 sein")
		}
		lr = *lrOverride
	}

	inputSize := len(n.weights[0][0])
	outputSize := len(n.weights[len(n.weights)-1])

	for idx, sample := range dataset {
		if len(sample.Input) != inputSize {
			return TrainResponse{}, fmt.Errorf("dataset[%d].input hat Größe %d statt %d", idx, len(sample.Input), inputSize)
		}
		if len(sample.Target) != outputSize {
			return TrainResponse{}, fmt.Errorf("dataset[%d].target hat Größe %d statt %d", idx, len(sample.Target), outputSize)
		}
	}

	indices := make([]int, len(dataset))
	for i := range indices {
		indices[i] = i
	}
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	lossHistory := make([]float64, 0, epochs)

	for epoch := 0; epoch < epochs; epoch++ {
		if shuffle {
			rng.Shuffle(len(indices), func(i, j int) {
				indices[i], indices[j] = indices[j], indices[i]
			})
		}

		epochLoss := 0.0

		for _, sampleIdx := range indices {
			sample := dataset[sampleIdx]

			as, zs, err := n.forwardWithCache(sample.Input)
			if err != nil {
				return TrainResponse{}, err
			}

			deltas := make([][]float64, len(n.weights))
			last := len(n.weights) - 1
			deltas[last] = make([]float64, len(n.weights[last]))

			for j := 0; j < len(n.weights[last]); j++ {
				diff := as[last+1][j] - sample.Target[j]
				epochLoss += 0.5 * diff * diff
				deltas[last][j] = diff * derivative(n.activations[last], zs[last][j], as[last+1][j])
			}

			for layer := len(n.weights) - 2; layer >= 0; layer-- {
				deltas[layer] = make([]float64, len(n.weights[layer]))
				for j := 0; j < len(n.weights[layer]); j++ {
					sum := 0.0
					for k := 0; k < len(n.weights[layer+1]); k++ {
						sum += n.weights[layer+1][k][j] * deltas[layer+1][k]
					}
					deltas[layer][j] = sum * derivative(n.activations[layer], zs[layer][j], as[layer+1][j])
				}
			}

			for layer := 0; layer < len(n.weights); layer++ {
				for j := 0; j < len(n.weights[layer]); j++ {
					for i := 0; i < len(n.weights[layer][j]); i++ {
						n.weights[layer][j][i] -= lr * deltas[layer][j] * as[layer][i]
					}
					n.biases[layer][j] -= lr * deltas[layer][j]
				}
			}
		}

		lossHistory = append(lossHistory, epochLoss/float64(len(dataset)))
	}

	finalLoss := lossHistory[len(lossHistory)-1]
	return TrainResponse{
		LossHistory: lossHistory,
		FinalLoss:   finalLoss,
		Samples:     len(dataset),
		Epochs:      epochs,
	}, nil
}

func computeMaxDeviationForNetwork(net *Network, dataset []Sample) (float64, error) {
	maxDeviation := 0.0

	for _, sample := range dataset {
		output, err := net.Forward(sample.Input)
		if err != nil {
			return 0, err
		}

		for i := 0; i < len(sample.Target); i++ {
			diff := math.Abs(output[i] - sample.Target[i])
			if diff > maxDeviation {
				maxDeviation = diff
			}
		}
	}

	return maxDeviation, nil
}

func newTrainerID() string {
	return fmt.Sprintf("trainer-%d-%d", time.Now().UnixNano(), rand.Int63())
}

func (t *Trainer) runLoop() {
	for {
		t.mu.Lock()
		shouldStop := t.stop
		if shouldStop {
			t.running = false
			t.mu.Unlock()
			return
		}

		res, err := t.net.Train(t.dataset, 1, t.learningRate, t.shuffle)
		if err != nil {
			t.lastErr = err.Error()
			t.running = false
			t.mu.Unlock()
			return
		}

		// TODO: prüfen, ob das überhaupt noch genutzt wird:
		deviation, err := computeMaxDeviationForNetwork(t.net, t.dataset)
		if err != nil {
			t.lastErr = err.Error()
			t.running = false
			t.mu.Unlock()
			return
		}

		t.epochsDone += 1
		t.lossHistory = append(t.lossHistory, res.FinalLoss)
		t.finalLoss = res.FinalLoss
		t.hasFinal = true
		t.deviation = deviation
		t.mu.Unlock()
	}
}

func getTrainer(trainerID string) (*Trainer, error) {
	trainersM.Lock()
	defer trainersM.Unlock()

	trainer, ok := trainers[trainerID]
	if !ok {
		return nil, fmt.Errorf("trainer_id nicht gefunden")
	}
	return trainer, nil
}

func decodeJSONArg[T any](args []js.Value, idx int, out *T) error {
	if len(args) <= idx {
		return fmt.Errorf("Argument %d fehlt", idx)
	}
	if args[idx].Type() != js.TypeString {
		return fmt.Errorf("Argument %d muss ein JSON-String sein", idx)
	}
	return json.Unmarshal([]byte(args[idx].String()), out)
}

func encodeJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf(`{"error":"JSON-Serialisierung fehlgeschlagen: %s"}`, err.Error())
	}
	return string(b)
}

func errJSON(err error) string {
	return encodeJSON(map[string]any{"error": err.Error()})
}

func createStateJS(_ js.Value, args []js.Value) any {
	var cfg NetworkConfig
	if err := decodeJSONArg(args, 0, &cfg); err != nil {
		return errJSON(err)
	}

	net, err := newNetwork(cfg)
	if err != nil {
		return errJSON(err)
	}

	return encodeJSON(map[string]any{"state": net.ToState()})
}

func forwardJS(_ js.Value, args []js.Value) any {
	var req ForwardRequest
	if err := decodeJSONArg(args, 0, &req); err != nil {
		return errJSON(err)
	}

	net, err := networkFromState(req.State)
	if err != nil {
		return errJSON(err)
	}

	out, err := net.Forward(req.Input)
	if err != nil {
		return errJSON(err)
	}

	return encodeJSON(ForwardResponse{Output: out})
}

func trainJS(_ js.Value, args []js.Value) any {
	var req TrainRequest
	if err := decodeJSONArg(args, 0, &req); err != nil {
		return errJSON(err)
	}

	net, err := networkFromState(req.State)
	if err != nil {
		return errJSON(err)
	}

	res, err := net.Train(req.Dataset, req.Epochs, req.LearningRate, req.Shuffle)
	if err != nil {
		return errJSON(err)
	}

	res.State = net.ToState()
	return encodeJSON(res)
}

func trainerInitJS(_ js.Value, args []js.Value) any {
	var req TrainerInitRequest
	if err := decodeJSONArg(args, 0, &req); err != nil {
		return errJSON(err)
	}

	if len(req.Dataset) == 0 {
		return errJSON(fmt.Errorf("dataset darf nicht leer sein"))
	}

	net, err := networkFromState(req.State)
	if err != nil {
		return errJSON(err)
	}

	trainerID := newTrainerID()
	trainer := &Trainer{
		id:           trainerID,
		net:          net,
		dataset:      append([]Sample(nil), req.Dataset...),
		learningRate: req.LearningRate,
		shuffle:      req.Shuffle,
		lossHistory:  []float64{},
	}

	trainersM.Lock()
	trainers[trainerID] = trainer
	trainersM.Unlock()

	return encodeJSON(map[string]any{"trainer_id": trainerID})
}

func trainerStartJS(_ js.Value, args []js.Value) any {
	var req TrainerCommandRequest
	if err := decodeJSONArg(args, 0, &req); err != nil {
		return errJSON(err)
	}

	trainer, err := getTrainer(req.TrainerID)
	if err != nil {
		return errJSON(err)
	}

	trainer.mu.Lock()
	if trainer.running {
		trainer.mu.Unlock()
		return encodeJSON(map[string]any{"ok": true})
	}

	trainer.stop = false
	trainer.running = true
	trainer.lastErr = ""
	trainer.mu.Unlock()

	go trainer.runLoop()

	return encodeJSON(map[string]any{"ok": true})
}

func trainerStatusJS(_ js.Value, args []js.Value) any {
	var req TrainerCommandRequest
	if err := decodeJSONArg(args, 0, &req); err != nil {
		return errJSON(err)
	}

	trainer, err := getTrainer(req.TrainerID)
	if err != nil {
		return errJSON(err)
	}

	trainer.mu.Lock()
	res := TrainerStatusResponse{
		TrainerID:    trainer.id,
		Running:      trainer.running,
		EpochsDone:   trainer.epochsDone,
		LossHistory:  append([]float64(nil), trainer.lossHistory...),
		FinalLoss:    trainer.finalLoss,
		HasFinalLoss: trainer.hasFinal,
		Deviation:    trainer.deviation,
		State:        trainer.net.ToState(),
		Error:        trainer.lastErr,
	}
	trainer.mu.Unlock()

	if res.Error != "" {
		return errJSON(fmt.Errorf(res.Error))
	}

	return encodeJSON(res)
}

func trainerStopJS(_ js.Value, args []js.Value) any {
	var req TrainerCommandRequest
	if err := decodeJSONArg(args, 0, &req); err != nil {
		return errJSON(err)
	}

	trainer, err := getTrainer(req.TrainerID)
	if err != nil {
		return errJSON(err)
	}

	trainer.mu.Lock()
	trainer.stop = true
	trainer.running = false
	trainer.mu.Unlock()

	return encodeJSON(map[string]any{"ok": true})
}

func trainerDisposeJS(_ js.Value, args []js.Value) any {
	var req TrainerCommandRequest
	if err := decodeJSONArg(args, 0, &req); err != nil {
		return errJSON(err)
	}

	trainersM.Lock()
	delete(trainers, req.TrainerID)
	trainersM.Unlock()

	return encodeJSON(map[string]any{"ok": true})
}

func listActivationsJS(_ js.Value, _ []js.Value) any {
	return encodeJSON(map[string]any{
		"activations": []string{"binary", "logistic", "relu"},
	})
}

func register(name string, fn func(js.Value, []js.Value) any) {
	cb := js.FuncOf(fn)
	callbacks = append(callbacks, cb)
	js.Global().Set(name, cb)
}

func main() {
	register("nnCreateState", createStateJS)
	register("nnForward", forwardJS)
	register("nnTrain", trainJS)
	register("nnListActivations", listActivationsJS)
	register("nnTrainerInit", trainerInitJS)
	register("nnTrainerStart", trainerStartJS)
	register("nnTrainerStatus", trainerStatusJS)
	register("nnTrainerStop", trainerStopJS)
	register("nnTrainerDispose", trainerDisposeJS)

	println("WASM bereit: nnCreateState, nnForward, nnTrain, nnListActivations, nnTrainerInit, nnTrainerStart, nnTrainerStatus, nnTrainerStop, nnTrainerDispose")

	select {}
}
