#!/bin/bash
# Setup script for ABSA evaluation framework

echo "=== ABSA Evaluation Framework Setup ==="
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p outputs
mkdir -p diverse_datasets
mkdir -p datasets

echo "Directories created: logs, outputs, diverse_datasets, datasets"
echo ""

# Check if original ABSA dataset exists
# Please update this path to point to your original ABSA dataset location
ORIGINAL_DATA_DIR="./datasets"

if [ -d "$ORIGINAL_DATA_DIR" ]; then
    echo "Found original ABSA datasets at: $ORIGINAL_DATA_DIR"
    echo ""
    
    # Option 1: Create symbolic links (recommended)
    echo "Option 1: Create symbolic links to original datasets (recommended)"
    echo "This will link to the original data without duplicating files."
    echo ""
    read -p "Create symbolic links? (y/n): " CREATE_SYMLINKS
    
    if [ "$CREATE_SYMLINKS" = "y" ] || [ "$CREATE_SYMLINKS" = "Y" ]; then
        echo "Creating symbolic links..."
        
        # Link data directory
        if [ ! -e "datasets/data" ]; then
            ln -s "$ORIGINAL_DATA_DIR/data" datasets/data
            echo "  - Linked datasets/data"
        else
            echo "  - datasets/data already exists, skipping"
        fi
        
        # Link few_shot directory
        if [ ! -e "datasets/few_shot" ]; then
            ln -s "$ORIGINAL_DATA_DIR/few_shot" datasets/few_shot
            echo "  - Linked datasets/few_shot"
        else
            echo "  - datasets/few_shot already exists, skipping"
        fi
        
        echo "Symbolic links created successfully!"
    else
        echo "Skipping symbolic links."
        echo "You can manually link or copy the datasets later:"
        echo "  ln -s $ORIGINAL_DATA_DIR/data datasets/data"
        echo "  ln -s $ORIGINAL_DATA_DIR/few_shot datasets/few_shot"
    fi
else
    echo "Original ABSA datasets not found at: $ORIGINAL_DATA_DIR"
    echo "Please update the ORIGINAL_DATA_DIR variable in this script"
    echo "or manually link/copy your datasets to the datasets/ directory."
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Set your OpenAI API key: export OPENAI_API_KEY='your-key'"
echo "3. Run example: python evaluate_llms.py --help"
echo "4. See example_usage.sh for more examples"
echo ""

