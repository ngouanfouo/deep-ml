def valid_palindrome(s):
    # s: a string of lowercase letters
    # return True or False
    def is_palindrome(lo, hi):
        while lo < hi:
            if s[lo] != s[hi]:
                return False
            lo += 1
            hi -= 1
        return True
    
    lo, hi = 0, len(s) - 1
    while lo < hi:
        if s[lo] == s[hi]:
            lo += 1
            hi -= 1
        else:
            # Mismatch: must delete either s[lo] or s[hi]
            return is_palindrome(lo + 1, hi) or is_palindrome(lo, hi - 1)
    return True