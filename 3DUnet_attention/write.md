Our loss, denoted by $\mathcal{L}_{focal}$ is defined as follows:
$$\mathcal{L}_{focal}=\sum_{(x)\in I, c\in C}w_{c}(x)f_{c}(x)$$

where $w_{c}(x)$ is the weight of the voxel $x$ associated with class $c$ computed based on the distance map applied on the ground truth $T_{c}$, and $f_{c}(x)$ is a function on the predicted probability of voxel $x$ belonging to class $c$. Formally, $w_{c}(x)$ and $f_{c}(x)$ are computed by the following equations:
$$w_{c}(x)=(L_{2}(x, G)+1)^{\alpha}\text{ where }\alpha\text{ is a constant scalar}\\$$

$$f_{c}(x)=\begin{cases}
    p_{c}(x)^{\beta} & \quad \text{if }x\text{ is not in }c\\
    (1-p_{c}(x))^{\beta} & \quad \text{otherwise}
\end{cases}$$

In our implementation, $\alpha$ is set to $-1$ so that 

Compared to offset curve loss, our loss applies distance function to the ground truth

In the training phase, we use a linear combination of Dice loss and our proposed loss to attain the advantages from both.

$$\mathcal{L}=(1-\lambda)\mathcal{L}_{Dice}+\lambda\mathcal{L}_{focal}$$
