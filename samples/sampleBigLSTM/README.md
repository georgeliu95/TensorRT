# NVIDIA TensorRT Sample "sampleBigLSTM"

By default, the sample will execute 8x smaller case.

To run BigLSTM case, e.g., batch 8:

    ./sample_big_lstm --weights=biglstm2x20x8.wts  --projected_size=1024 --state_size=8192 --batch=8

At the very end median execution time is reported.  As well as estimate of useful bandwidth and useful flops.

    ./sample_big_lstm --help

    Optional params:
      --datadir=<path>        Path string to math up to 10 levels from the current dir (default = data/samples/big-lstm/)
      --weights=<file>        Name of the file with weights, located in the datadir (default = toybiglstm2x20.wts)
      --batch=N               Set batch size (default = 1)
      --num_layers=N          Set number of LSTMP layers (default = 2)
      --num_steps=N           Set number of LSTMP time steps (default = 20)
      --state_size=N          Set number of  LSTM memory cells (default = 512)
      --projected_size=N      Set size of LSTMP projection layer (default = 128)
      --device=N              Set cuda device to N (default = 0)
      --iterations=N          Run N iterations (default = 12)
      --avgRuns=N             Set avgRuns to N - perf is measured as an average of avgRuns (default=10)
      --workspace=N           Set workspace size in megabytes (default = 31)
      --half2                 Run in paired fp16 mode (default = false)
      --int8                  Run in int8 mode (default = false)
      --verbose               Use verbose logging (default = false)
      --sm                    Enable output Softmax layer (default = false)
      --perplexity            Perplexity calculation from all time steps
      --overrideGPUClocks=val Override GPU clocks for %SOL calculation, in GHz
      --overrideMemClocks=val Override memory clocks for %SOL calculation, in GHz
      --overrideBWRatio=val   Override practical/theoretical peak bandwidth ratio for %SOL calculation
      --useDLACore=N          Specify a DLA core for layers that support DLA. Value can range from 0 to N-1, where N is the number of DLA cores on the platform.

