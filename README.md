#### 数组遍历框架

```cpp
void traverse(vector<int>& arr) {
    for (int i = 0; i < arr.size(); i++) {
        // 迭代访问 arr[i]
    }
}
```

#### 链表遍历框架

```cpp
/* 基本的单链表节点 */
class ListNode {
    public:
        int val;
        ListNode* next;
};

void traverse(ListNode* head) {
    for (ListNode* p = head; p != nullptr; p = p->next) {
        // 迭代访问 p->val
    }
}

void traverse(ListNode* head) {
    // 递归访问 head->val
    traverse(head->next);
}
```

#### 二叉遍历框架

```cpp
void traverse(TreeNode* root) {
    if (root == nullptr) {
        return;
    }
    // 前序位置
    traverse(root->left);
    // 中序位置
    traverse(root->right);
    // 后序位置
}
```

#### N叉树遍历框架

```cpp
/* 基本的 N 叉树节点 */
class TreeNode {
public:
    int val;
    vector<TreeNode*> children;
};

void traverse(TreeNode* root) {
    for (TreeNode* child : root->children)
        traverse(child);
}
```

#### 双指针解链表

1. 单链表的中点，[链表的中间结点](https://leetcode.cn/problems/middle-of-the-linked-list/)
2. 判断链表是否包含环   [环形链表 ](https://leetcode.cn/problems/linked-list-cycle-ii/)

#### 双指针解数组(一般有序的数组，一定要想到双指针)

1. 删除重复项   [删除有序数组中的重复项](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/)
2. 移除元素   [移除元素](https://leetcode.cn/problems/remove-element/)
3. 移动零    [移动零](https://leetcode.cn/problems/move-zeroes/)
4. 反转数组    [反转字符串](https://leetcode.cn/problems/reverse-string/)
5. 回文串判断    [最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/)

#### 层序遍历框架

```cpp
// 输入一棵二叉树的根节点，层序遍历这棵二叉树
void levelTraverse(TreeNode* root) {
    if (root == nullptr) return;
    queue<TreeNode*> q;
    q.push(root);

    // 从上到下遍历二叉树的每一层
    while (!q.empty()) {
        int sz = q.size();
        // 从左到右遍历每一层的每个节点
        for (int i = 0; i < sz; i++) {
            TreeNode* cur = q.front();
            q.pop();
            // 将下一层节点放入队列
            if (cur->left != nullptr) {
                q.push(cur->left);
            }
            if (cur->right != nullptr) {
                q.push(cur->right);
            }
        }
    }
}
```

#### 动态规划框架

常见的背包问题，找零钱问题一般用动态框架

```python
# 自顶向下递归的动态规划
def dp(状态1, 状态2, ...):
    for 选择 in 所有可能的选择:
        # 此时的状态已经因为做了选择而改变
        result = 求最值(result, dp(状态1, 状态2, ...))
    return result

# 自底向上迭代的动态规划
# 初始化 base case
dp[0][0][...] = base case
# 进行状态转移
for 状态1 in 状态1的所有取值：
    for 状态2 in 状态2的所有取值：
        for ...
            dp[状态1][状态2][...] = 求最值(选择1，选择2...)

```

1. 例如解零钱的代码:    [322. 零钱兑换](https://leetcode.cn/problems/coin-change/)

```cpp
int coinChange(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, amount + 1);
    // 数组大小为 amount + 1，初始值也为 amount + 1
    dp[0] = 0;

    // 外层 for 循环在遍历所有状态的所有取值
    for (int i = 0; i < dp.size(); i++) {
        // 内层 for 循环在求所有选择的最小值
        for (int coin : coins) {
            // 子问题无解，跳过
            if (i - coin < 0) {
                continue;
            }
            dp[i] = min(dp[i], 1 + dp[i - coin]);
        }
    }
    return (dp[amount] == amount + 1) ? -1 : dp[amount];
}
```

2. 最长递增子序列问题：[最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/)

```cpp
int lengthOfLIS(vector<int>& nums) {
    // 定义：dp[i] 表示以 nums[i] 这个数结尾的最长递增子序列的长度
    vector<int> dp(nums.size(), 1);
    // base case：dp 数组全都初始化为 1
    for (int i = 0; i < nums.size(); i++) {
        for (int j = 0; j < i; j++) {
            if (nums[i] > nums[j]) 
                dp[i] = max(dp[i], dp[j] + 1);
        }
    }
    
    int res = 0;
    for (int i = 0; i < dp.size(); i++) {
        res = max(res, dp[i]);
    }
    return res;
}
```

3. 背包问题

```cpp
#include <cassert>

// 完全背包问题
// 输入：
// - W: 背包的最大容量
// - N: 物品的数量
// - wt: 物品的重量数组，wt[i-1]表示第i个物品的重量
// - val: 物品的价值数组，val[i-1]表示第i个物品的价值
// 输出：
// - 最大价值
int knapsack(int W, int N, int wt[], int val[]) {
    assert(N == sizeof(wt)/sizeof(wt[0])); // 确保N和wt的长度匹配
    int dp[N + 1][W + 1]; // base case 已初始化为0
    for (int i = 1; i <= N; i++) {
        for (int w = 1; w <= W; w++) {
            if (w - wt[i - 1] < 0) {
                // 这种情况下只能选择不装入背包
                dp[i][w] = dp[i - 1][w];
            } else {
                // 装入或者不装入背包，择优
                dp[i][w] = std::max(
                    dp[i - 1][w - wt[i-1]] + val[i-1], 
                    dp[i - 1][w]
                );
            }
        }
    }   
    return dp[N][W];
}
```

#### 回溯算法框架

回溯算法一般用于解排列问题

```python
result = []
def backtrack(路径, 选择列表):
    if 满足结束条件:
        result.add(路径)
        return
    
    for 选择 in 选择列表:
        做选择
        backtrack(路径, 选择列表)
        撤销选择
```

##### [子集](https://leetcode.cn/problems/subsets/)（元素无重不可复选）

```cpp
class Solution {
private:
    vector<vector<int>> res;
    // 记录回溯算法的递归路径
    vector<int> track;

public:
    // 主函数
    vector<vector<int>> subsets(vector<int>& nums) {
        backtrack(nums, 0);
        return res;
    }

    // 回溯算法核心函数，遍历子集问题的回溯树
    void backtrack(vector<int>& nums, int start) {

        // 前序位置，每个节点的值都是一个子集
        res.push_back(track);

        // 回溯算法标准框架
        for (int i = start; i < nums.size(); i++) {
            // 做选择
            track.push_back(nums[i]);
            // 通过 start 参数控制树枝的遍历，避免产生重复的子集
            backtrack(nums, i + 1);
            // 撤销选择
            track.pop_back();
        }
    }
};
```

这里需要注意到的是, 回溯算法里的start是主渐渐+1的。 而循环里的i是start。

##### [组合](https://leetcode.cn/problems/combinations/)(元素无重不可复选)

```cpp
class Solution {
public:
    vector<vector<int>> res;
    // 记录回溯算法的递归路径
    deque<int> track;

    // 主函数
    vector<vector<int>> combine(int n, int k) {
        backtrack(1, n, k);
        return res;
    }

    void backtrack(int start, int n, int k) {
        // base case
        if (k == track.size()) {
            // 遍历到了第 k 层，收集当前节点的值
            res.push_back(vector<int>(track.begin(), track.end()));
            return;
        }

        // 回溯算法标准框架
        for (int i = start; i <= n; i++) {
            // 选择
            track.push_back(i);
            // 通过 start 参数控制树枝的遍历，避免产生重复的子集
            backtrack(i + 1, n, k);
            // 撤销选择
            track.pop_back();
        }
    }
};
```

##### [全排列](https://leetcode.cn/problems/permutations/)(元素无重不可复选)

```cpp
class Solution {
public:
    // 存储所有排列结果的列表
    vector<vector<int>> res;
    // 记录回溯算法的递归路径
    list<int> track;
    // 标记数字使用状态的数组，0 表示未被使用，1 表示已被使用
    bool* used;

    /* 主函数，输入一组不重复的数字，返回它们的全排列 */
    vector<vector<int>> permute(vector<int>& nums) {
        used = new bool[nums.size()]();
        // 满足回溯框架时需要添加 bool 类型默认初始化为 false
        backtrack(nums);
        return res;
    }

    // 回溯算法核心函数
    void backtrack(vector<int>& nums) {
        // base case，到达叶子节点
        if (track.size() == nums.size()) {
            // 收集叶子节点上的值
            res.push_back(vector<int>(track.begin(), track.end()));
            return;
        }
        // 回溯算法标准框架
        for (int i = 0; i < nums.size(); i++) {
            // 已经存在 track 中的元素，不能重复选择
            if (used[i]) {
                continue;
            }
            // 做选择
            used[i] = true;
            track.push_back(nums[i]);
            // 进入下一层回溯树
            backtrack(nums);
            // 取消选择
            track.pop_back();
            used[i] = false;
        }
    }
};

```

1. 我们用 `used` 数组标记已经在路径上的元素避免重复选择，然后收集所有叶子节点上的值，就是所有全排列的结果。  
2. 我们可以看到的是这里的`i`是从0开始，而不再是start。

##### [子集 II](https://leetcode.cn/problems/subsets-ii/)(元素可重不可复选)

```cpp
class Solution {
    vector<vector<int>> res; // 输出结果
    vector<int> track; // 搜索路径
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end()); // 排序，让相同的元素靠在一起
        backtrack(nums, 0);
        return res; // 返回结果
    }

    void backtrack(vector<int>& nums, int start) { // start 为当前的枚举位置
        res.emplace_back(track); // 前序位置，每个节点的值都是一个子集
        for(int i = start; i < nums.size(); i++) {
            if (i > start && nums[i] == nums[i - 1]) { // 剪枝逻辑，值相同的相邻树枝，只遍历第一条
                continue;
            }
            track.emplace_back(nums[i]); // 添加至路径
            backtrack(nums, i + 1); // 进入下一层决策树
            track.pop_back(); // 回溯
        }
    }
};
```

##### [全排列 II](https://leetcode.cn/problems/permutations-ii/)(元素可重不可复选)

```cpp
class Solution {
public:
    // 保存结果
    vector<vector<int>> res;
    // 记录当前位置的元素
    vector<int> track;
    // 记录元素是否被使用
    vector<bool> used;

    // 主函数
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        // 排序，让相同的元素靠在一起
        sort(nums.begin(), nums.end());
        // 初始化used数组
        used = vector<bool>(nums.size(), false);
        // 回溯
        backtrack(nums);
        // 返回结果
        return res;
    }

    // 回溯函数
    void backtrack(vector<int>& nums) {
        // 当长度相等时，将结果记录
        if (track.size() == nums.size()) {
            res.push_back(track);
            return;
        }

        // 遍历没有被使用过的元素
        for (int i = 0; i < nums.size(); i++) {
            if (used[i]) {
                continue;
            }
            // 新添加的剪枝逻辑，固定相同的元素在排列中的相对位置
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) {
                continue;
            }
            // 添加元素，标记为使用过
            track.push_back(nums[i]);
            used[i] = true;
            // 继续回溯
            backtrack(nums);
            // 回溯
            track.pop_back();
            used[i] = false;
        }
    }
};
```

##### 子集/组合(元素无重复,可复选)  [组合总和](https://leetcode.cn/problems/combination-sum/)

```cpp
class Solution {
public:
    vector<vector<int>> res;
    // 记录回溯的路径
    deque<int> track;
    // 记录 track 中的路径和
    int trackSum = 0;

    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        backtrack(candidates, 0, target);
        return res;
    }

    // 回溯算法主函数
    void backtrack(vector<int>& nums, int start, int target) {
        // base case，找到目标和，记录结果
        if (trackSum == target) {
            res.push_back(vector<int>(track.begin(), track.end()));
            return;
        }
        // base case，超过目标和，停止向下遍历
        if (trackSum > target) {
            return;
        }
        // 回溯算法标准框架
        for (int i = start; i < nums.size(); i++) {
            // 选择 nums[i]
            trackSum += nums[i];
            track.push_back(nums[i]);
            // 递归遍历下一层回溯树
            // 同一元素可重复使用，注意参数
            backtrack(nums, i, target);
            // 撤销选择 nums[i]
            trackSum -= nums[i];
            track.pop_back();
        }
    }
};
```

##### 排列(元素无重可复选) 

```cpp
class Solution {
public:
    vector<vector<int>> res;
    deque<int> track;

    vector<vector<int>> permuteUnique(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        backtrack(nums);
        return res;
    }

    void backtrack(vector<int>& nums) {
        if (track.size() == nums.size()) {
            res.push_back(vector(track.begin(), track.end()));
            return;
        }

        for (int i = 0; i < nums.size(); i++) {
            //剪枝操作，判断当前节点是否已经在track中
            if (i > 0 && nums[i] == nums[i - 1] && find(track.begin(), track.end(), nums[i - 1]) != track.end()) {
                continue;
            }
            track.push_back(nums[i]);
            backtrack(nums);
            track.pop_back();
        }
    }
};
```

这个代码相较于前面的排列代码少了used。

#### 二分搜索框架

```cpp
int binarySearch(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;

    while(left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) {
            ...
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid - 1;
        }
    }
    return ...;
}

```

 `left + (right - left) / 2` 就和 `(left + right) / 2` 的结果相同，但是有效防止了 `left` 和 `right` 太大，直接相加导致溢出的情况。

左侧边界

```cpp
while (left < right) {
    //...
}
// 如果索引越界，说明数组中无目标元素，返回 -1
if (left < 0 || left >= nums.length) {
    return -1;
}
// 判断一下 nums[left] 是不是 target
return nums[left] == target ? left : -1;

```

右侧边界

```cpp
// 最后改成返回 left - 1
    if (left - 1 < 0 || left - 1 >= nums.size()) {
        return -1;
    }
```

#### 滑动窗口

**滑动窗口算法技巧主要用来解决子数组问题，比如让你寻找符合某个条件的最长/最短子数组**。

```cpp
/* 滑动窗口算法框架 */
void slidingWindow(string s) {
    // 用合适的数据结构记录窗口中的数据，根据具体场景变通
    // 比如说，我想记录窗口中元素出现的次数，就用 map
    // 我想记录窗口中的元素和，就用 int
    unordered_map<char, int> window;
    
    int left = 0, right = 0;
    while (right < s.size()) {
        // c 是将移入窗口的字符
        char c = s[right];
        window.add(c)
        // 增大窗口
        right++;
        // 进行窗口内数据的一系列更新
        ...

        /*** debug 输出的位置 ***/
        // 注意在最终的解法代码中不要 print
        // 因为 IO 操作很耗时，可能导致超时
        printf("window: [%d, %d)\n", left, right);
        /********************/
        
        // 判断左侧窗口是否要收缩
        while (left < right && window needs shrink) {
            // d 是将移出窗口的字符
            char d = s[left];
            window.remove(d)
            // 缩小窗口
            left++;
            // 进行窗口内数据的一系列更新
            ...
        }
    }
}
```

例题[找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/)

```cpp
vector<int> findAnagrams(string s, string t) {
    unordered_map<char, int> need, window;
    for (char c : t) need[c]++;

    int left = 0, right = 0;
    int valid = 0;
    vector<int> res; // 记录结果
    while (right < s.size()) {
        char c = s[right];
        right++;
        // 进行窗口内数据的一系列更新
        if (need.count(c)) {
            window[c]++;
            if (window[c] == need[c]) 
                valid++;
        }
        // 判断左侧窗口是否要收缩
        while (right - left >= t.size()) {
            // 当窗口符合条件时，把起始索引加入 res
            if (valid == need.size())
                res.push_back(left);
            char d = s[left];
            left++;
            // 进行窗口内数据的一系列更新
            if (need.count(d)) {
                if (window[d] == need[d])
                    valid--;
                window[d]--;
            }
        }
    }
    return res;
}
```

#### 数组小技巧

1. 前缀和

   前缀和技巧适用于快速、频繁地计算一个索引区间内的元素

   ```cpp
   class NumArray {
       private:
           // 前缀和数组
           vector<int> preSum;
       
       public:
           /* 输入一个数组，构造前缀和 */
           NumArray(vector<int>& nums) {
               // preSum[0] = 0，便于计算累加和
               preSum.resize(nums.size() + 1);
               // 计算 nums 的累加和
               for (int i = 1; i < preSum.size(); i++) {
                   preSum[i] = preSum[i - 1] + nums[i - 1];
               }
           }
           
           /* 查询闭区间 [left, right] 的累加和 */
           int sumRange(int left, int right) {
               return preSum[right + 1] - preSum[left];
           }
   };
   ```

   典型例题: [二维区域和检索 - 矩阵不可变](https://leetcode.cn/problems/range-sum-query-2d-immutable/)

   ```cpp
   class NumMatrix {
   public:
       vector<vector<int>> preSum;
       NumMatrix(vector<vector<int>>& matrix) {
           int m = matrix.size(), n = matrix[0].size();
           if (m == 0 || n == 0)
               return;
           // 构造前缀和矩阵
           preSum = vector<vector<int>>(m + 1, vector<int>(n + 1));
           for (int i = 1; i <= m; i++)
               for (int j = 1; j <= n; j++) {
                   preSum[i][j]=preSum[i][j-1]+preSum[i-1][j]+matrix[i-1][j-1]-preSum[i-1][j-1];
               }
       }
   
       int sumRegion(int row1, int col1, int row2, int col2) {
           return preSum[row2+1][col2+1]-preSum[row1][col2+1]-preSum[row2+1][col1]+preSum[row1][col1];
       }
   };
   ```

2. 差分数组

   差分数组的主要使用场景是频繁对原始数组的某个区间的元素进行增减

   ```cpp
   int diff[nums.size()];
   // 构造差分数组
   diff[0] = nums[0];
   for (int i = 1; i < nums.size(); i++) {
       diff[i] = nums[i] - nums[i - 1];
   }
   ```

   我们也可以通过差分数组反向求得原数组.

   ```cpp
   int res[diff.size()];
   // 根据差分数组构造结果数组
   res[0] = diff[0];
   for (int i = 1; i < diff.size(); i++) {
       res[i] = res[i - 1] + diff[i];
   }
   ```

   **这样构造差分数组 `diff`，就可以快速进行区间增减的操作**，如果你想对区间 `nums[i..j]` 的元素全部加 3，那么只需要让 `diff[i] += 3`，然后再让 `diff[j+1] -= 3` 即可：

3. 二维的矩阵

   1. 特殊的旋转方式，先对角线，而后再外圈旋转

   2. 确定上下左右四个边界

      ```cpp
      #include <vector>
      #include <deque>
      
      using namespace std;
      
      vector<int> spiralOrder(vector<vector<int>>& matrix) {
          int m = matrix.size(), n = matrix[0].size();
          int upper_bound = 0, lower_bound = m - 1;
          int left_bound = 0, right_bound = n - 1;
          vector<int> res;
          // res.size() == m * n 则遍历完整个数组
          while (res.size() < m * n) {
              if (upper_bound <= lower_bound) {
                  // 在顶部从左向右遍历
                  for (int j = left_bound; j <= right_bound; j++) {
                      res.push_back(matrix[upper_bound][j]);
                  }
                  // 上边界下移
                  upper_bound++;
              }
              
              if (left_bound <= right_bound) {
                  // 在右侧从上向下遍历
                  for (int i = upper_bound; i <= lower_bound; i++) {
                      res.push_back(matrix[i][right_bound]);
                  }
                  // 右边界左移
                  right_bound--;
              }
              
              if (upper_bound <= lower_bound) {
                  // 在底部从右向左遍历
                  for (int j = right_bound; j >= left_bound; j--) {
                      res.push_back(matrix[lower_bound][j]);
                  }
                  // 下边界上移
                  lower_bound--;
              }
              
              if (left_bound <= right_bound) {
                  // 在左侧从下向上遍历
                  for (int i = lower_bound; i >= upper_bound; i--) {
                      res.push_back(matrix[i][left_bound]);
                  }
                  // 左边界右移
                  left_bound++;
              }
          }
          return res;
      }
      ```

#### 田忌赛马(优先队列问题)[优势洗牌](https://leetcode.cn/problems/advantage-shuffle/)

```cpp
vector<int> advantageCount(vector<int>& nums1, vector<int>& nums2) {
    int n = nums1.size();
    // 给 nums2 降序排序
    priority_queue<pair<int, int>> maxpq;
    for (int i = 0; i < n; i++) {
        maxpq.emplace(i, nums2[i]);
    }

    // 给 nums1 升序排序
    sort(nums1.begin(), nums1.end());

    // nums1[left] 是最小值，nums1[right] 是最大值
    int left = 0, right = n - 1;
    vector<int> res(n);

    while (!maxpq.empty()) {
        auto [i, maxval] = maxpq.top();
        maxpq.pop();
        if (maxval < nums1[right]) {
            // 如果 nums1[right] 能胜过 maxval，那就自己上
            res[i] = nums1[right];
            right--;
        } else {
            // 否则用最小值混一下，养精蓄锐
            res[i] = nums1[left];
            left++;
        }
    }
    return res;
}
```

#### 图问题

1. 构图

   ```cpp
    vector<vector<int>> buildGraph(int numCourse,
                                      vector<vector<int>>& prerequisites) {
           vector<vector<int>> graph = vector<vector<int>>(numCourse);
           for (vector<int> edge : prerequisites) {
               graph[edge[0]].push_back(edge[1]);
           }
           return graph;
       }
   ```

   构造的这个图是以邻接表的形式存在的。

2. 遍历图：

   ```cpp
     void traverse(vector<vector<int>> &graph, int s) {
           if (onPath[s]) {
               hasCycle = true;
           }
           if (visited[s] || hasCycle) {
               return;
           }
           visited[s] = true;
           onPath[s] = true;
           for (int t : graph[s]) {
               traverse(graph, t);
           }
           onPath[s] = false;
       }
   ```

   这里的这个`onPath[s]`记录的是路径，而这个`visted[s]` 记录你已经访问过的点， 可以减少再次访问，避免减少性能。

3. 前期准备工作

   ```cpp
    vector<bool> visited;
       vector<bool> onPath;
       bool hasCycle = false;
       bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
           vector<vector<int>> graph = buildGraph(numCourses, prerequisites);
           visited = vector<bool>(numCourses, false);
           onPath = vector<bool>(numCourses, false);
           for (int i = 0; i < numCourses; i++)
               traverse(graph, i);
           return !hasCycle;
       }
   ```

   例题有: [课程表](https://leetcode.cn/problems/course-schedule/)，名流问题。

#### 岛屿问题

框架

```cpp
void traverse(TreeNode* root) {
    traverse(root->left);
    traverse(root->right);
}

void dfs(vector<vector<int>>& grid, int i, int j, vector<vector<bool>>& visited) {
    int m = grid.size(), n = grid[0].size();
    if (i < 0 || j < 0 || i >= m || j >= n) {
        return;
    }
    if (visited[i][j]) {
        return;
    }
    visited[i][j] = true;
    dfs(grid, i - 1, j, visited); // 上
    dfs(grid, i + 1, j, visited); // 下
    dfs(grid, i, j - 1, visited); // 左
    dfs(grid, i, j + 1, visited); // 右
}
```

#### 常用的位操作

1. **利用或操作 `|` 和空格将英文字符转换为小写**

```java
('a' | ' ') = 'a'
('A' | ' ') = 'a'
```

1. **利用与操作 `&` 和下划线将英文字符转换为大写**

```java
('b' & '_') = 'B'
('B' & '_') = 'B'
```

1. **利用异或操作 `^` 和空格进行英文字符大小写互换**

```java
('d' ^ ' ') = 'D'
('D' ^ ' ') = 'd'
```

以上操作能够产生奇特效果的原因在于 ASCII 编码。ASCII 字符其实就是数字，恰巧空格和下划线对应的数字通过位运算就能改变大小写。有兴趣的读者可以查 ASCII 码表自己算算，本文就不展开讲了。

1. **不用临时变量交换两个数**

```java
int a = 1, b = 2;
a ^= b;
b ^= a;
a ^= b;
// 现在 a = 2, b = 1
```

1. **加一**

```java
int n = 1;
n = -~n;
// 现在 n = 2
```

1. **减一**

```java
int n = 2;
n = ~-n;
// 现在 n = 1
```

1. **判断两个数是否异号**

```java
int x = -1, y = 2;
boolean f = ((x ^ y) < 0); // true

int x = 3, y = 2;
boolean f = ((x ^ y) < 0); // false
```

如果说前 6 个技巧的用处不大，这第 7 个技巧还是比较实用的，利用的是**补码编码**的符号位。整数编码最高位是符号位，负数的符号位是 1，非负数的符号位是 0，再借助异或的特性，可以判断出两个数字是否异号。

##### n&(n-1)的运用

1. 位1的个数

   ```cpp
   int hammingWeight(int n) {
       int res = 0;
       while (n != 0) {
           n = n & (n - 1);
           res++;
       }
       return res;
   }
   ```

2. 2的幂

   ```cpp
   bool isPowerOfTwo(int n) {
       if (n <= 0) return false;
       return (n & (n - 1)) == 0;
   }
   ```

   ##### a^a=0的运用

   一个数和它本身做异或运算结果为 0，即 `a ^ a = 0`；一个数和 0 做异或运算的结果为它本身，即 `a ^ 0 = a`。

   1. 只出现一次的数字

      ```cpp
      int singleNumber(vector<int>& nums) {
          int res = 0;
          for (int n : nums) {
              res ^= n;
          }
          return res;
      }
      ```

#### 丑数(只包含质因数2,3,5)

```cpp
class Solution {
public:
    /**
     * 判断一个数是否为丑数
     * @param n 要判断的数
     * @return 如果 n 是丑数返回 true，否则返回 false
     */
    bool isUgly(int n) {
        if (n <= 0) return false;
        // 如果 n 是丑数，分解因子应该只有 2, 3, 5
        while (n % 2 == 0) n /= 2;
        while (n % 3 == 0) n /= 3;
        while (n % 5 == 0) n /= 5;
        // 如果能够成功分解，说明是丑数
        return n == 1;
    }
};
```

#### 接雨水(采用双指针)[42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/)

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        int left = 0, right = height.size() - 1;
        int l_max = 0, r_max = 0;

        int res = 0;
        while (left < right) {
            l_max = max(l_max, height[left]);
            r_max = max(r_max, height[right]);

            // res += min(l_max, r_max) - height[i]
            if (l_max < r_max) {
                res += l_max - height[left];
                left++;
            } else {
                res += r_max - height[right];
                right--;
            }
        }
        return res;
    }
};

```



[11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/)

```cpp
class Solution {
public:
    int maxArea(vector<int>& height) {
        int left = 0, right = height.size() - 1;
        int res = 0;
        while (left < right) {
            // [left, right] 之间的矩形面积
            int cur_area = min(height[left], height[right]) * (right - left);
            res = max(res, cur_area);
            // 双指针技巧，移动较低的一边
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return res;
    }
};
```

#### 解决括号相关性问题

```cpp
bool visted(string str){
    stack<char>s;
    for(int i=0;i<str.length();i++){
       char c=str[i];
        if(c=='{'||c=='['||c=='(')
            s.push(c);
        else if(!s.empty()&&leftOf(c)==s.top())
            s.pop();
        eles
            return false;
    }
    return true;
}
char leftOf(char c) {
    if (c == '}') return '{';
    if (c == ')') return '(';
    return '[';
}
```

# -
