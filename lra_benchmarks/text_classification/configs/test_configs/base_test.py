# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base Configuration."""

from lra_benchmarks.text_classification.configs import base_tc_config


def get_config():
  """Get the default hyperparameter configuration."""
  config = base_tc_config.get_config()
  config.batch_size = 4
  config.eval_frequency = 10
  config.num_train_steps = 20
  config.num_heads = 2
  config.num_layers = 2
  config.num_data_entries = {'train': 1000, 'valid': 100, 'test': 100}
  return config