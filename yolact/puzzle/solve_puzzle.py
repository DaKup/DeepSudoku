import numpy as np


def find_empty_location(arr, l):
    inds = np.where(arr == 0)
    
    if (inds[0].size == 0):
        return False
    
    l[0] = inds[0][0]
    l[1] = inds[1][0]
    
    return True


def used_in_row(arr, row, num):
    return num in arr[row]


def used_in_col(arr, col, num):
    return num in arr[:, col]


def used_in_box(arr, row, col, num):
    return num in arr[row : row + 3, col : col + 3]


def check_location_is_safe(arr, row, col, num):
    return not (used_in_row(arr, row, num) or used_in_col(arr, col, num) or used_in_box(arr, row - row % 3, col - col % 3, num))


def solve_sudoku(arr):
    # 'l' is a list variable that keeps the record of row and col in find_empty_location Function
    l = [0, 0]
    
    # If there is no unassigned location, we are done
    if (not find_empty_location(arr, l)):
        return True
    
    # Assigning list values to row and col that we got from the above Function
    row = l[0]
    col = l[1]
    
    # consider digits 1 to 9
    for num in range(1, 10):
        # if looks promising
        if (check_location_is_safe(arr, row, col, num)):
            # make tentative assignment
            arr[row, col] = num
            
            # return, if success, ya!
            if (solve_sudoku(arr)):
                return True
            
            # failure, unmake & try again
            arr[row, col] = 0
    
    # this triggers backtracking
    return False


def is_solvable(arr):
    for i in range(9):
        r = arr[i, :]
        r = r[np.where(r != 0)]
        
        if (r.size != np.unique(r).size):
            return False
        
        c = arr[:, i]
        c = c[np.where(c != 0)]
        
        if (c.size != np.unique(c).size):
            return False
        
        b = arr[3 * (i // 3) : 3 * (i // 3) + 3, 3 * (i % 3) : 3 * (i % 3) + 3]
        b = b[np.where(b != 0)]
        
        if (b.size != np.unique(b).size):
            return False
    
    return True


def main():
    # assigning values to the grid 
    grid = np.array(
       [[0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 8, 5],
        [0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    
    if (is_solvable(grid)):
        solve_sudoku(grid)
        print(grid)


if __name__=="__main__":
    main()

