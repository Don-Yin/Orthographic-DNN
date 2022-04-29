import platform

# from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from utils.evaluate.cosine_similarity import compute_cosine_similarity


class Evaluate:
    """_summary_
    Input: model + which word + prime_types + which layer
    Output: cosine similarity
    e.g.,

    pnasnet = timm.create_model("pnasnet5large", pretrained=True, num_classes=1000)
    evaluate = Evaluate(model=pnasnet)
    evaluate.main(word="abduct", prime_types=("ID", "SN-F"), which_layer="penultimate")

    returns cosine similarity

    """

    def __init__(self, model, word: str, prime_types: tuple[str], which_layer: str, prime_data):
        self.model = model
        self.word = word
        self.prime_types = prime_types
        self.which_layer = which_layer
        self.prime_data = prime_data
        self.get_image_tensor()

    def get_image_tensor(self):
        image_label_prime: list[list] = [
            i[0].replace(".png", "").split(("/" if platform.system() != "Windows" else "\\"))[2::]
            for i in self.prime_data.imgs
        ]
        image_indices: tuple[int] = (
            image_label_prime.index([self.word, self.prime_types[0]]),
            image_label_prime.index([self.word, self.prime_types[1]]),
        )
        self.tensors = (
            self.prime_data[image_indices[0]][0].unsqueeze(0).cuda().detach(),
            self.prime_data[image_indices[1]][0].unsqueeze(0).cuda().detach(),
        )

    def compute_similarity_node_wise(self):
        if self.which_layer == "classification":
            outputs = (self.model(self.tensors[0]), self.model(self.tensors[1]))
            similarity = compute_cosine_similarity(outputs)
            return similarity

        elif self.which_layer == "penultimate":
            which_layer = -3
            nodes, _ = get_graph_node_names(self.model)
            feature_extractor = create_feature_extractor(model=self.model, return_nodes=[nodes[which_layer]])
            outputs = (
                feature_extractor(self.tensors[0])[nodes[which_layer]],
                feature_extractor(self.tensors[1])[nodes[which_layer]],
            )
            similarity = compute_cosine_similarity(outputs)
            return similarity

    def compute_similarity_layer_wise(self):
        if self.which_layer == "classification":
            outputs = (self.model(self.tensors[0]), self.model(self.tensors[1]))
            similarity = compute_cosine_similarity(outputs)
            return similarity

        elif self.which_layer == "penultimate":
            self.activation = {}
            self.detach_tensors = True
            all_layers = self.group_all_layers()
            all_layers_names = []
            which_layer = -2
            hook_lists = []

            for idx, i in enumerate(all_layers):
                name = "{}: {}".format(idx, str.split(str(i), "(")[0])
                all_layers_names.append(name)
                hook_lists.append(i.register_forward_hook(self.get_activation(name)))

            self.model(self.tensors[0])
            output_0 = self.activation[all_layers_names[which_layer]].flatten().unsqueeze(0)
            self.model(self.tensors[1])
            output_1 = self.activation[all_layers_names[which_layer]].flatten().unsqueeze(0)
            similarity = compute_cosine_similarity((output_0, output_1))
            return similarity
        
        elif self.which_layer == "penultimate_visualizer":
            self.activation = {}
            self.detach_tensors = True
            all_layers = self.group_all_layers()
            all_layers_names = []
            which_layer = -2
            hook_lists = []

            for idx, i in enumerate(all_layers):
                name = "{}: {}".format(idx, str.split(str(i), "(")[0])
                all_layers_names.append(name)
                hook_lists.append(i.register_forward_hook(self.get_activation(name)))

            self.model(self.tensors[0])
            output_0 = self.activation[all_layers_names[which_layer]]
            self.model(self.tensors[1])
            output_1 = self.activation[all_layers_names[which_layer]]
            return output_0, output_1

        elif self.which_layer == "all":
            self.activation = {}
            self.detach_tensors = True
            all_layers = self.group_all_layers()
            all_layers_names = []
            hook_lists = []

            for idx, i in enumerate(all_layers):
                name = "{}: {}".format(idx, str.split(str(i), "(")[0])
                all_layers_names.append(name)
                hook_lists.append(i.register_forward_hook(self.get_activation(name)))

            self.model(self.tensors[0])
            intermediate_activations_1 = [self.activation[i].flatten().unsqueeze(0) for i in self.activation.keys()]
            self.model(self.tensors[1])
            intermediate_activations_2 = [self.activation[i].flatten().unsqueeze(0) for i in self.activation.keys()]
            similarities = [
                compute_cosine_similarity((intermediate_activations_1[i], intermediate_activations_2[i]))
                for i in range(len(self.activation))
            ]

            return similarities

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach() if self.detach_tensors else output

        return hook

    def group_all_layers(self):
        all_layers = []

        def recursive_group(model):
            for layer in model.children():
                if not list(layer.children()):
                    all_layers.append(layer)
                else:
                    recursive_group(layer)

        recursive_group(self.model)
        return all_layers
