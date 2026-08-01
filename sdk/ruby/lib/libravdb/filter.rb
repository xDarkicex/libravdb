module LibraVDB
  class Filter
    attr_reader :type, :args

    def initialize(type, args)
      @type = type
      @args = args
    end

    def self.eq(field, value)
      new("eq", { field: field, value: value })
    end

    def self.neq(field, value)
      new("neq", { field: field, value: value })
    end

    def self.gt(field, value)
      new("gt", { field: field, value: value })
    end

    def self.gte(field, value)
      new("gte", { field: field, value: value })
    end

    def self.lt(field, value)
      new("lt", { field: field, value: value })
    end

    def self.lte(field, value)
      new("lte", { field: field, value: value })
    end

    def self.in(field, value)
      new("in", { field: field, value: value })
    end

    def self.contains(field, value)
      new("contains", { field: field, value: value })
    end

    def self.and(*filters)
      new("and", filters.map(&:to_hash))
    end

    def self.or(*filters)
      new("or", filters.map(&:to_hash))
    end

    def self.not(filter)
      new("not", filter.to_hash)
    end

    def and(other)
      Filter.and(self, other)
    end

    def or(other)
      Filter.or(self, other)
    end

    def to_hash
      { type: @type }.merge(@args)
    end
  end
end
