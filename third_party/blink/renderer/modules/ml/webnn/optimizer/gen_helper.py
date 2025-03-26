import re


def get_optype_list(graph_mojom_source):
    optype_list = []

    idx = 0
    for i in range(len(graph_mojom_source)):
        if 'union Operation {' in graph_mojom_source[i]:
            idx = i
            break

    for i in range(idx+1, len(graph_mojom_source)):
        if '}' in graph_mojom_source[i]:
            break
        line = graph_mojom_source[i].strip()
        if line.startswith('//'):
            continue
        if line.endswith(';'):
            line = line[:-1]
            optype_list.append(line.split()[0])

    return optype_list


class Member:
    def __init__(self) -> None:
        optional = False
        type = ''
        name = ''
        default_value = ''


class OpTemplate:
    def __init__(self):
        self.op_type = ''
        self.op_inputs = []
        self.op_outputs = []
        self.attributes = []


struct_line_pattern = r"^struct\s+(\w+)\s*\{"
enum_line_pattern = r"^enum\s+(\w+)\s*\{"


def extract_variable_info(line):
    pattern = r"(\S+(?:<[^>]+>)?\??)\s+(\w+)\s*(?:=\s*([^\s;]+))?\s*;"

    match = re.match(pattern, line)

    if match:
        type_, name, default_value = match.groups()
        return type_, name, default_value if default_value else ''
    else:
        return None


op_nested_struct_table = {}
op_nested_enum_table = {}

# current idx should be the line of struct like: struct Slice {


def eat_op_struct(graph_mojom_source, idx, struct_name):
    struct = OpTemplate()
    struct.op_type = struct_name
    left_braces_budget = 1
    idx += 1
    while left_braces_budget > 0:
        line = graph_mojom_source[idx]
        if '{' in line:
            line = line.strip()
            match_struct = re.search(struct_line_pattern, line)
            match_enum = re.search(enum_line_pattern, line)
            if match_struct:
                nested_struct_name = match_enum.group(1)
                if nested_struct_name not in op_nested_struct_table.keys():
                    op_nested_enum_table[struct_name] = [nested_struct_name]
                else:
                    op_nested_enum_table[struct_name].append(
                        nested_struct_name)
            if match_enum:
                nested_enum_name = match_enum.group(1)
                if nested_enum_name not in op_nested_enum_table.keys():
                    op_nested_enum_table[struct_name] = [nested_enum_name]
                else:
                    op_nested_enum_table[struct_name].append(nested_enum_name)

            left_braces_budget += 1
            idx += 1
            continue
        if '}' in line:
            left_braces_budget -= 1
            idx += 1
            continue
        # not in a nested struct
        if left_braces_budget == 1:
            line = line.strip()
            result = extract_variable_info(line)
            if result:
                type_, name, default_value = result
                member = Member()

                if type_.endswith('?'):
                    member.optional = True
                    type_ = type_[:-1]
                member.type = type_
                member.name = name
                member.default_value = default_value

                if 'output_operand_id' in name:
                    struct.op_outputs.append(member)
                elif 'operand_id' in name:
                    struct.op_inputs.append(member)
                else:
                    struct.attributes.append(member)

        idx += 1

    return idx, struct


def get_op_templates(graph_mojom_source, optypes):
    op_templates = []
    idx = 0
    while idx < len(graph_mojom_source):
        line = graph_mojom_source[idx]
        match = re.search(struct_line_pattern, line)
        if match:
            struct_name = match.group(1)
            if struct_name in optypes:
                idx, struct = eat_op_struct(
                    graph_mojom_source, idx, struct_name)
                op_templates.append(struct)
                continue
            else:
                idx += 1
                continue
        else:
            idx += 1
            continue
    return op_templates


def op_template_to_cpp(op_template):
    input_num = 0
    output_num = 0
    # check input number

    if len(op_template.op_inputs) > 1:
        input_num = len(op_template.op_inputs)
    else:
        assert (len(op_template.op_inputs) == 1)
        if 'array' in op_template.op_inputs[0].type:
            input_num = 'kDynamicIOCount'
        else:
            input_num = 1

    assert len(op_template.op_outputs) == 1
    op_output = op_template.op_outputs[0]
    if op_output.name == 'output_operand_id':
        output_num = 1
    else:
        assert op_output.name == 'output_operand_ids'
        output_num = 'kDynamicIOCount'

    cpp_class_name = f'{op_template.op_type}Node'

    cpp_input_getter = ''

    idx = -1
    for ip in op_template.op_inputs:
        idx += 1
        if ip.type == 'uint64':
            ip_name = ip.name[:-11]
            cpp_input_getter += f"Node* {ip_name}() const {{ return GetInputNode({idx}); }}\n"

        else:
            assert ip.type == 'array<uint64>'
            # do nothing if it is array<uint64>

    cpp_attributes = ''

    for attr in op_template.attributes:
        if attr.name == 'label':
            continue

        dtype = attr.type
        match = re.search(r"array<(\w+)", dtype)
        is_array = False
        if match:
            is_array = True
            dtype = match.group(1)
        cpptype = ''
        if dtype == 'int64':
            cpptype = 'int64_t'

        elif dtype == 'int32':
            cpptype = 'int32_t'

        elif dtype == 'uint64':
            cpptype = 'uint64_t'

        elif dtype == 'uint32':
            cpptype = 'uint32_t'

        elif dtype == 'float':
            cpptype = 'float'

        elif dtype == 'bool':
            cpptype = 'bool'

        elif dtype == 'string':
            cpptype = 'WTF::String'

        elif op_template.op_type in op_nested_enum_table.keys() and dtype in op_nested_enum_table[op_template.op_type]:
            # nested enum
            cpptype = f'webnn::mojom::blink::{op_template.op_type}::{dtype}'
        else:
            # gloabl struct
            cpptype = f'webnn::mojom::blink::{dtype}'

        if is_array:
            cpptype = f'Vector<{cpptype}>'

        cpp_attributes += f'{cpptype} {attr.name};\n'

    cpp_trace_code = f'void Trace(Visitor* visitor) const override {{ Node::Trace(visitor); }}\n'
    op_kind_getter_code = f'OpKind op_kind() const override  {{ return OpKind::k{op_template.op_type}; }}'
    static_op_kind_getter_code = f'static OpKind StaticOpKind() {{ return OpKind::k{op_template.op_type}; }}'

    hpp_code = f'''
    class {cpp_class_name} : public NodeT<{cpp_class_name},{input_num},{output_num}> {{
        public:
            {static_op_kind_getter_code}
            {cpp_class_name}() = default;
            {cpp_input_getter}
            {cpp_attributes}
            {cpp_trace_code}
            {op_kind_getter_code}

    }};

    '''

    return hpp_code


if __name__ == '__main__':
    graph_mojom_path = 'G:/jianxiao/chromium/src/services/webnn/public/mojom/webnn_graph.mojom'

    graph_mojom_source = ''

    with open(graph_mojom_path, 'r') as f:
        graph_mojom_source = f.readlines()

    optypes = get_optype_list(graph_mojom_source)
    print(f' /*{optypes}*/')
    op_templates = get_op_templates(graph_mojom_source, optypes)

    # define spec ops
    V_spec_ops = ''
    for op_type in optypes:
        V_spec_ops += f'V({op_type}) \\\n'
    V_spec_ops = V_spec_ops[:-3]

    cpp_code = rf'''
        #define WEBNN_OPTIMIZER_SPEC_OPERATION_LIST \
        {V_spec_ops}

    '''
    print(cpp_code)
    # opset
    print('// code for opset')
    cpp_opsets = ''
    for op_type in optypes:
        cpp_opsets += f'k{op_type},\n'

    cpp_code = f'''
    enum class OpKind {{
    // spec ops
        {cpp_opsets}
    }};
    '''
    print(cpp_code)

    print('\n\n\n\n')

    # op def
    print('// code for op def')
    for op_template in op_templates:
        cpp_code = op_template_to_cpp(op_template)
        print(cpp_code)