
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        fib_sequence = [0, 1]
        for i in range(2, n):
            next_term = fib_sequence[-1] + fib_sequence[-2]
            fib_sequence.append(next_term)
        return fib_sequence

def main():
    terms = 10
    result = fibonacci(terms)
    print(f"Fibonacci sequence up to {terms} terms: {result}")

if __name__ == "__main__":
    main()