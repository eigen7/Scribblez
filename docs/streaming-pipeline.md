# Streaming pipeline

I'd like to do a major overhaul of how the model generation/training pipeline works.

Right now, we run `generate_data.py`, which runs the c++ binary to produce game logs and writes to
disk. Then, we run `train.py` to run epochs of training over those game logs.

This pipeline may be useful in the future, when we're using an expensive process to generate games.

When we're doing hasty-vs-hasty, however, game generation is SO fast, that I think this deserves a
specialized pipeline. Because game generation is so fast, there doesn't seem to me to be much point
in recycling positions across training epochs. Might as well just constantly generate fresh games.

Since we're only sampling once per game, **we don't need to randomly shuffle our training data**!
This means that our c++ game-generation process can avoid writing to disk altogether. Instead, it
can just serialize the game on the fly and directly write to tensor `float*` addresses passed to it
from the python side.

The python side can have a size-N ring buffer of tensor objects, and can pass these into the c++
process at startup. The c++ launches game-threads, and as each game completes, a writer writes the
game to a row of the current ring-buffer item. When the current item is filled up, it notifies the
python, which can then proceed with a train step. After finishing its train step, it notifies the
c++ to let it know that that ring-buffer item can be released. If the c++ has circled around the
ring buffer and the next item hasn't been released, it should pause game production until the slot
is available. Similarly, if the python finishes processing a minibatch but the next ring-buffer
item has not yet been marked as processed by the c++, it should wait until that notification
arrives.

I expect N=2 should suffice. This should nicely saturate compute resources. The training loop on
the python side (NN eval + backprop) is GPU-heavy and CPU-light. The parallel game generation on the
c++ side is CPU-heavy and does not use the GPU. With N=2, we can consistently have both running at
the same time, saturating GPU and CPU concurrently.

# Python script

I envision a single python script, launched via something like:

```
./py/train_post_move_model.py -t tag_name
```

This creates a directory like `/workspace/mount/post_move_training/<tag_name>`, where stuff is
written. Note that without the need to write game data to disk, this directory should be much more
lightweight than our current `generate_data.py` + `train.py` pipeline.

It should launch the web dashboard that `py/dashboard.py` currently produces. But we should modify
the dashboard to reflect the streaming nature of this pipeline. Meaning, it should show real-time
performance metrics, like showing the game throughput rate, and showing backpressure stats to inform
us on where the bottleneck is (i.e., is the c++ waiting for the python, or is the python waiting for
the c++?). The data for these plots should be saved to persistent sqlite3 tables, to support
restarts.

On the first run of the script, it should generate a validation set, which we periodically measure
the model against. This validation set *does* need to be written to disk, so that we have
consistency across restarts. The dashboard should include standard validation metrics, like
train vs test loss, test set accuracy.

A single model checkpoint file can be reused constantly. We can export an onnx model according to
a configured cadence (after training over P positions).

# Potential c++ performance improvements

If the c++ is the bottleneck, we have some potential tricks up our sleeve. We can implement MAGPIE's
WMP word-list + shadow-play. This should significantly increase the c++ game throughput rate. Note
that this only works when the equity evaluation is achieved through static-leave-lookups, as is done
by hasty. No need to look into this yet - only do so once we find evidence that the c++ is the
bottleneck.

# Testing

Every component that gets written should have appropriate tests, to the extent possible. This
includes the c++, the python, and the web front-end. Test early and often.
