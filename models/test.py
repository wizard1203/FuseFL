# def outer_function():
#     shared_variable = 0

#     def inner_function():
#         nonlocal shared_variable
#         shared_variable += 1
#         return shared_variable

#     return inner_function

# my_function = outer_function()
# print(my_function())
# print(my_function()) 


# def outer_function():
#     shared_variable = 0

#     def inner_function():
#         nonlocal shared_variable
#         shared_variable += 1
#         return shared_variable
#     return inner_function()


# print(outer_function())
# print(outer_function())





# def outer_function():
#     shared_variable = 0

#     def inner_function():
#         nonlocal shared_variable
#         shared_variable += 1
#         return shared_variable
#     print(inner_function())
#     print(inner_function())



# outer_function()




# def map_values(A, range_end):
#     # Calculate the total number of elements to distribute
#     total_elements = range_end + 1
#     num_values = len(A)
    
#     # Calculate the number of elements to assign to each value in A
#     elements_per_value = total_elements // num_values

#     # Handle any remaining elements that don't fit evenly
#     remaining_elements = total_elements % num_values

#     # Initialize the mapping dictionary
#     mapping = {}
#     current_index = 0

#     for value in A:
#         # Calculate the number of elements for this value
#         num_elements = elements_per_value + (1 if remaining_elements > 0 else 0)
#         remaining_elements -= 1

#         # Map each element in the range to the current value
#         for i in range(current_index, current_index + num_elements):
#             mapping[i] = value

#         # Update the index for the next value
#         current_index += num_elements

#     return mapping

# # List A
# A = [8, 16, 32, 64]

# # Create the mapping for the range 0-17
# mapping = map_values(A, 17)
# print(mapping)




a = list(range(10))
b = a[1:]
print(b)



a = list(range(10))
b = a[1:6]
print(b)




a = list(range(10))
b = a[6:10]
print(b)









