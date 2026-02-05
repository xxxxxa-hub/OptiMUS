Imagine a bustling city where seven primary communication stations are in operation. These stations, ranging from a central source (Station 0) to the main endpoint (Station 6), are interconnected by a network of telecommunication lines with varying transfer capacities.

Here are the capacities of each connection, representing the maximum amount of data (in terabytes) that can be transferred from one station to another:

- From Station 0 (Source): Can send data to Station 1 (17TB), Station 2 (3TB), Station 3 (20TB), Station 4 (11TB), Station 5 (11TB), and Station 6 (3TB).
- From Station 1: Can send data to Station 0 (11TB), Station 2 (12TB), Station 3 (18TB), Station 4 (5TB), Station 5 (6TB), and Station 6 (4TB).
- From Station 2: Can send data to Station 0 (7TB), Station 1 (2TB), Station 3 (15TB), Station 4 (13TB), Station 5 (20TB), and Station 6 (18TB).
- From Station 3: Can send data to Station 0 (7TB), Station 1 (4TB), Station 2 (2TB), Station 4 (2TB), Station 5 (15TB), and Station 6 (3TB).
- From Station 4: Can send data to Station 0 (19TB), Station 1 (12TB), Station 2 (16TB), Station 3 (4TB), Station 5 (19TB), and Station 6 (4TB).
- From Station 5: Can send data to Station 0 (13TB), Station 1 (20TB), Station 2 (10TB), Station 3 (14TB), Station 4 (15TB), and Station 6 (1TB).
- From Station 6 (Endpoint): Can send data back to Station 0 (3TB), Station 1 (3TB), Station 2 (6TB), Station 3 (20TB), Station 4 (9TB), and Station 5 (10TB).

In this scenario, a capacity of "0" indicates there is no direct telecommunication link between the two respective stations.

Your challenge is to determine the maximum amount of data that can be transferred from Station 0 (the source) to Station 6 (the endpoint) per hour, while ensuring the data flow along each telecommunication line does not exceed its set capacity. What is the maximum amount of data, in terabytes, that can be transferred from the source to the endpoint within one hour?