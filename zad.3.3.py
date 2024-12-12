import heapq

def dijkstra(graph, start):

    priority_queue = []
    heapq.heappush(priority_queue, (0, start))

    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    predecessors = {node: None for node in graph}

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)


        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, predecessors

graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('C', 2), ('D', 6)],
    'C': [('A', 4), ('B', 2), ('D', 3)],
    'D': [('B', 6), ('C', 3)]
}

start_node = 'A'
distances, predecessors = dijkstra(graph, start_node)

print("Najkrótsze odległości:", distances)
print("Poprzednicy:", predecessors)
