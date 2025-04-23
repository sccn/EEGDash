config = {
  "required_fields": ["data_name"],
  "attributes": {
    "data_name": "str",
    "dataset": "str",
    "bidspath": "str",
    "subject": "str",
    "task": "str",
    "session": "str",
    "run": "str",
    "sampling_frequency": "float",
    "modality": "str",
    "nchans": "int",
    "ntimes": "int"
  },
  "description_fields": ["subject", "session", "run", "task", "age", "gender", "sex"],
  "bids_dependencies_files": [
    "dataset_description.json", 
    "participants.tsv", 
    "events.tsv", 
    "events.json", 
    "eeg.json", 
    "electrodes.tsv", 
    "channels.tsv", 
    "coordsystem.json"
  ],
  "accepted_query_fields": ["data_name", "dataset"]
}