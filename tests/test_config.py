from concatenative.config import DEFAULT_CONFIG, deep_merge, load_config
from pathlib import Path
import pytest
import copy

@pytest.fixture
def temp_config_file(tmp_path: Path):

    def _create_file(content: str) -> str:
        config_path = tmp_path / 'test_config.toml'
        config_path.write_text(content)
        return str(config_path)
    
    return _create_file

@pytest.fixture
def nested_dictionary():
    nested_dict = { 
            'segmentation': {
                'max_duration_s': 0.2,
                'onset': {
                    'hop_length': 512,
                    'backtrack': True,
                    'normalise': False,
                }
            }, 
            'features': {
                'hop_length': 512, 
                'pitch': {
                    'fmin': 50,
                    'fmax': 5000
            }
        }
    }

    return nested_dict

class TestDeepMerge():

    def test_empty_dictionaries_returns_empty(self):
        '''
        Test passing in empty dictionaries for both 
        target and destination produces empty dictionary
        '''

        out_config = deep_merge({}, {})
        assert len(out_config) == 0


    def test_empty_dest_flat_target(self):
        '''
        Test passing in empty destination dict and
        flat target dict returns a dictionary
        identical to target 
        '''

        flat_target = {'key': 'value'}
        out_config = deep_merge({}, flat_target)

        assert flat_target == out_config

    def test_empty_dest_nested_target(self, nested_dictionary):
        '''
        Test passing in empty destination dict and
        nested target dict returns a dictionary
        identical to target 
        '''

        out_config = deep_merge({}, nested_dictionary)
        assert nested_dictionary == out_config


    def test_default_dest_empty_target(self, nested_dictionary):
        '''
        Test passing in default destination dict and
        empty target dict returns a dictionary
        identical to dest 
        '''
        
        out_config = deep_merge(nested_dictionary, {})
        assert nested_dictionary == out_config 

    
    def test_target_overwrites_dest_value(self, nested_dictionary):
        '''
        Test that a value in the destination is overwritten by 
        a value in the target
        '''

        target = copy.deepcopy(nested_dictionary)
        target['segmentation']['onset']['hop_length'] = 2048

        out_config = deep_merge(nested_dictionary, target)

        assert target == out_config

    
    def test_missing_key_from_target(self, nested_dictionary):
        '''
        Test that if a key is missing from the target
        the value from the destination is retained
        '''

        target = copy.deepcopy(nested_dictionary)
        del target['segmentation']['onset']

        out_config = deep_merge(nested_dictionary, target)
        assert out_config == nested_dictionary


    def test_extra_key_in_target_added(self, nested_dictionary):
        '''
        Test that if a key is in the target but not the destination
        it gets added to the destination
        '''

        destination = copy.deepcopy(nested_dictionary)
        del destination['features']

        out_config = deep_merge(destination, nested_dictionary)
        assert out_config == nested_dictionary
        
    def test_merge_strategy(self, nested_dictionary):
        '''
        Test that two dictionaries are merged together
        '''

        expected = {
            'segmentation': {
                'max_duration_s': 0.2,
                'min_duration_s': 0.05,
                'onset': {
                    'hop_length': 512,
                    'backtrack': True,
                    'normalise': False,
                },
                'none': {
                    'max_snippets': 3,
                    'remove_silent': True
                }
            }, 
            'features': {
                'hop_length': 512, 
                'pitch': {
                    'fmin': 50,
                    'fmax': 5000
                }
            }
        }

        destination = copy.deepcopy(nested_dictionary)
        del destination['features']
        destination['segmentation']['none'] = {
            'max_snippets': 3,
            'remove_silent': True
        }

        target = copy.deepcopy(nested_dictionary)
        del target['segmentation']['onset']

        target['segmentation']['min_duration_s'] = 0.05

        out_config = deep_merge(destination, target)
        assert expected == out_config


class TestLoadConfig():

    def test_loads_defaults_when_no_path_given(self):
        out_config = load_config()
        assert out_config == DEFAULT_CONFIG

    def test_fallback_to_default_if_file_not_found(self):
        out_config = load_config(Path('a/made/up/path'))
        assert out_config == DEFAULT_CONFIG

    def test_merges_user_config_over_defaults(self, temp_config_file):

        injected_toml_content = """
        [segmentation.onsets]
        hop_length = 2048
        """

        config_path = Path(temp_config_file(injected_toml_content))
        config = load_config(config_path)

        # Check override
        assert config['segmentation']['onsets']['hop_length'] == 2048

        # Check missing keys have retained defaults
        assert config['segmentation']['onsets']['backtrack'] == False
        assert config['segmentation']['onsets']['normalise'] == False

    def test_handles_malformed_toml(self, temp_config_file):

        injected_toml_content = """
        Invalid toml = ???\n[bad
        """

        config_path = Path(temp_config_file(injected_toml_content))
        config = load_config(config_path)

        assert config == DEFAULT_CONFIG