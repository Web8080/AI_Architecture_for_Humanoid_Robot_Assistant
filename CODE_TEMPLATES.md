# Code Templates for Humanoid Robot Assistant

## Author Information

All code files should include the following author information:

- **Author**: Victor Ibhafidon
- **Organization**: Xtainless Technologies
- **Email**: info@xtainlesstech.com

---

## Python Template

```python
"""
[Module Name] - [Brief Description]

[Detailed description of what this module does]

Author: Victor Ibhafidon
Organization: Xtainless Technologies
Email: info@xtainlesstech.com
Date: YYYY-MM-DD
License: MIT
"""

import logging
from typing import Optional, List, Dict, Any

# Module code here
```

---

## Python Class Template

```python
"""
[Module Name] - [Brief Description]

Author: Victor Ibhafidon
Organization: Xtainless Technologies
Date: YYYY-MM-DD
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConfigClass:
    """Configuration for [ClassName]"""
    param1: str
    param2: int = 10


class ClassName:
    """
    [Brief class description]
    
    This class provides functionality for [purpose].
    
    Attributes:
        attribute1: Description
        attribute2: Description
    
    Example:
        >>> obj = ClassName(config)
        >>> result = obj.method()
    """
    
    def __init__(self, config: ConfigClass):
        """
        Initialize [ClassName].
        
        Args:
            config: Configuration object
        """
        self.config = config
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def method(self, param: str) -> Dict[str, Any]:
        """
        [Method description]
        
        Args:
            param: Parameter description
        
        Returns:
            Dictionary containing results
        
        Raises:
            ValueError: If param is invalid
        """
        # Implementation
        pass


# ==================== Testing ====================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test code here
    config = ConfigClass(param1="test")
    obj = ClassName(config)
    print("Tests passed!")
```

---

## C++ Template (for ROS2 nodes)

```cpp
/**
 * [Module Name] - [Brief Description]
 *
 * [Detailed description]
 *
 * Author: Victor Ibhafidon
 * Organization: Xtainless Technologies
 * Email: info@xtainlesstech.com
 * Date: YYYY-MM-DD
 * License: MIT
 */

#include <iostream>
#include <memory>
#include <string>

// ROS2 includes
#include "rclcpp/rclcpp.hpp"

namespace robot_brain {

class NodeName : public rclcpp::Node {
public:
    explicit NodeName(const std::string& name) 
        : Node(name) {
        RCLCPP_INFO(this->get_logger(), "Node initialized");
    }

private:
    // Members
};

}  // namespace robot_brain

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<robot_brain::NodeName>("node_name");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
```

---

## YAML Configuration Template

```yaml
# ==============================================================================
# [Configuration Name]
# 
# Author: Victor Ibhafidon
# Organization: Xtainless Technologies
# Date: YYYY-MM-DD
# ==============================================================================

module_name:
  parameter1: value1
  parameter2: value2
  
# Add configuration sections here
```

---

## Dockerfile Template

```dockerfile
# ==============================================================================
# Dockerfile for [Service Name]
# 
# Author: Victor Ibhafidon
# Organization: Xtainless Technologies
# Date: YYYY-MM-DD
# ==============================================================================

FROM base_image:tag

LABEL maintainer="Victor Ibhafidon <info@xtainlesstech.com>"
LABEL description="[Service description]"
LABEL version="1.0.0"

# Build steps here
```

---

## Shell Script Template

```bash
#!/bin/bash
# ==============================================================================
# [Script Name] - [Brief Description]
#
# Author: Victor Ibhafidon
# Organization: Xtainless Technologies
# Date: YYYY-MM-DD
# ==============================================================================

set -e  # Exit on error

# Script implementation
```

---

## Jupyter Notebook Template

```python
# %% [markdown]
# # [Notebook Title]
# 
# **Author**: Victor Ibhafidon  
# **Organization**: Xtainless Technologies  
# **Date**: YYYY-MM-DD
#
# ## Overview
# [Brief description of what this notebook does]
#
# ## Objectives
# - Objective 1
# - Objective 2

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
%matplotlib inline
%load_ext autoreload
%autoreload 2

# %% [markdown]
# ## Main Analysis

# %%
# Code here
```

---

## README Template

```markdown
# [Module/Component Name]

**Author**: Victor Ibhafidon  
**Organization**: Xtainless Technologies  
**Last Updated**: YYYY-MM-DD

## Overview

[Brief description]

## Features

- Feature 1
- Feature 2

## Installation

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage

\`\`\`python
from module import Class

# Example usage
obj = Class()
result = obj.method()
\`\`\`

## API Reference

See [API documentation](link).

## License

MIT License - Copyright (c) 2025 Victor Ibhafidon, Xtainless Technologies
```

---

## Documentation Style Guide

### Docstring Format (Google Style)

```python
def function_name(param1: str, param2: int = 10) -> Dict[str, Any]:
    """
    Brief description of what the function does.
    
    Longer description providing more context about the function's
    purpose, behavior, and any important notes.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter (default: 10)
    
    Returns:
        Dictionary containing:
            - key1: Description of key1
            - key2: Description of key2
    
    Raises:
        ValueError: If param1 is empty
        RuntimeError: If operation fails
    
    Example:
        >>> result = function_name("test", 20)
        >>> print(result["key1"])
        'value1'
    
    Note:
        Any additional notes or warnings about usage.
    """
    pass
```

---

## Git Commit Message Format

```
type(scope): Brief description (50 chars max)

Longer explanation if needed (wrap at 72 chars)

- Bullet points for details
- Another point

Author: Victor Ibhafidon
Refs: #issue-number
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Example**:
```
feat(nlp): Add emotion detection to dialogue manager

Implemented 7-way emotion classification using DistilBERT
fine-tuned on GoEmotions dataset.

- Added EmotionDetector class
- Integrated with dialogue manager
- Added unit tests

Author: Victor Ibhafidon
Refs: #42
```

---

## File Naming Conventions

- **Python modules**: `snake_case.py` (e.g., `intent_classifier.py`)
- **Classes**: `PascalCase` (e.g., `IntentClassifier`)
- **Functions/methods**: `snake_case` (e.g., `predict_intent`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`)
- **Config files**: `kebab-case.yaml` (e.g., `system-config.yaml`)

---

## Copyright Notice

For files requiring explicit copyright:

```
Copyright (c) 2025 Victor Ibhafidon, Xtainless Technologies

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

**Maintained by**: Victor Ibhafidon, Xtainless Technologies

