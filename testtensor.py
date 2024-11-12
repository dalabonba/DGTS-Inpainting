import torch

# tensor = torch.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
tensor = torch.tensor([[1, 2, 3]])
print("tensor")
print(tensor.shape)
print(tensor)

print("---------------")
repeat_tensor = torch.repeat_interleave(tensor, repeats=2, dim=0)
print("repeat_tensor")
print(repeat_tensor.shape)
print(repeat_tensor)


print("---------------")
cat_tensor = torch.cat([tensor, tensor], dim=0)
print("cat_tensor")
print(cat_tensor.shape)
print(cat_tensor)
