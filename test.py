import pyperclip

text = r"""
\begin{tikzcd}
W_{FPP} \arrow[d]                    & {W_{FPP}, T_{TL-12}} \arrow[d]        &                                                                                 &                                                                                                                       &                   \\
\text{abduct} \arrow[rd, bend right] & \text{abductTL-12} \arrow[d]          &                                                                                 & {S_C(\text{feature map}_{W_{abduct}}, \text{feature map}_{W_{abduct, TL-12}})} \arrow[r]                              & kendall_init_demo \\
                                     & \text{network input matrix} \arrow[r] & \text{network penultimate layer} \arrow[r] \arrow[ru, bend right, shift left=2] & \text{network output layer}                                                                                           &                   \\
\text{abound} \arrow[rd, bend right] & \text{aboundTL-12} \arrow[d]          &                                                                                 & {S_C(\text{feature map}_{W_{abduct}}, \text{feature map}_{W_{abound, TL-12}})} \arrow[ruu, bend right, shift right]   &                   \\
                                     & \text{network input matrix} \arrow[r] & \text{network penultimate layer} \arrow[r] \arrow[ru, bend right, shift left=2] & \text{network output layer}                                                                                           &                   \\
\text{abrupt} \arrow[rd, bend right] & \text{abruptTL-12} \arrow[d]          &                                                                                 & {S_C(\text{feature map}_{W_{abduct}}, \text{feature map}_{W_{abrupt, TL-12}})} \arrow[ruuuu, bend right, shift right] &                   \\
                                     & \text{network input matrix} \arrow[r] & \text{network penultimate layer} \arrow[r] \arrow[ru, bend right, shift left=2] & \text{network output layer}                                                                                           &                   \\
                                     & ...                                   & ...                                                                             & ...                                                                                                                   &                  
\end{tikzcd}"""


tran = {
    r"\text{network input matrix}": r"\vcenter{\hbox{\includegraphics[scale=0.3]{network_component/input.png}}}",
    r"\text{network intermediate layers}": r"\vcenter{\hbox{\includegraphics[scale=0.3]{network_component/intermediate.png}}}",
    r"\text{network penultimate layer}": r"\vcenter{\hbox{\includegraphics[scale=0.3]{network_component/intermediate_penultimate.png}}}",
    r"\text{network output layer}": r"\vcenter{\hbox{\includegraphics[scale=0.3]{network_component/output.png}}}",
    r"\text{abduct}": r"\vcenter{\hbox{\includegraphics[scale=0.2]{dig/abduct_ID.png}}}",
    r"\text{abound}": r"\vcenter{\hbox{\includegraphics[scale=0.2]{dig/abound_ID.png}}}",
    r"\text{abrupt}": r"\vcenter{\hbox{\includegraphics[scale=0.2]{dig/abrupt_ID.png}}}",
    r"\text{abductTL-12}": r"\vcenter{\hbox{\includegraphics[scale=0.2]{dig/abduct_TL-12.png}}}",
    r"\text{aboundTL-12}": r"\vcenter{\hbox{\includegraphics[scale=0.2]{dig/abound_TL-12.png}}}",
    r"\text{abruptTL-12}": r"\vcenter{\hbox{\includegraphics[scale=0.2]{dig/abrupt_TL-12.png}}}",
    r"kendall_init_demo": r"\input{diagrams/kendalls_init_demo}",
}
for i in tran:
    text = text.replace(i, tran[i])
pyperclip.copy(text)
