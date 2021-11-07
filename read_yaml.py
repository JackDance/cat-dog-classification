import yaml


def parse_yaml(yaml_path):
    with open (yaml_path, 'r', encoding='utf-8') as f:
        # content = f.read()
        content = yaml.safe_load(f)
        # print(f1['data_root']) # key-->value
        return content

# yaml_path = './config.yaml'
# parse_yaml(yaml_path)

if __name__ == '__main__':
    yaml_path = './config.yaml'
    f1 = parse_yaml(yaml_path)
    print(f1['data_root'])




